import os
import json
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, BartTokenizerFast, BartForConditionalGeneration
from transformers.trainer_utils import set_seed
from utils.load_dataset import Text2SQLDataset
from tokenizers import AddedToken
from test_suite.Evaluator import RestaurantsEvaluator, AdvisingEvaluator, WikiSQLEvaluator, IMDBEvaluator

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 128,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')
    parser.add_argument('--table_file_path', type = str, default = "data/wikisql/dev.tables.jsonl",
                        help = 'file path of table file.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--output", type = str, default = "predicted_pql.txt",
                help = "save file of the predicted sqls.")
    parser.add_argument("--dataset", type = str, default = "advising",
                        help = "dataset name")
    
    opt = parser.parse_args()

    return opt

def read_data(file_path='error_case.txt'):
    ground_truth = []
    predicted = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for i in range(0, len(lines)-1, 3):  # 以步长3来读取每一组数据
            gt_line = lines[i].strip().replace("ground_truth: ", "")
            pd_line = lines[i+1].strip().replace("predicted: ", "")
            ground_truth.append(gt_line)
            predicted.append(pd_line)

    return ground_truth, predicted

def decode_pqls(
    db_path,
    generator_outputs,
    batch_db_ids,
    tokenizer,
    dataset,
    table_file_path = None
):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_sqls = []
    if dataset == 'advising':
        evaluator = AdvisingEvaluator(db_path)
    elif dataset == 'restaurants':
        evaluator = RestaurantsEvaluator(db_path)
    elif dataset == 'wikisql':
        evaluator = WikiSQLEvaluator(db_path, table_file_path)
    elif dataset == 'imdb':
        evaluator = IMDBEvaluator(db_path)
    else:
        raise ValueError("Invalid dataset name")

    
    for batch_id in range(batch_size):
        pred_executable_sql = "sql placeholder"
        db_id = batch_db_ids[batch_id]

        for seq_id in range(num_return_sequences):
            # pred_sequence是结果
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens = True)
            pred_sql = pred_sequence.split("|")[-1].strip()
            pred_sql = pred_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
            pred_executable_sql = pred_sql
            
            try:
                if dataset == 'wikisql':
                    r = evaluator.execution(pred_executable_sql, db_id)
                else:
                    r = evaluator.execution(pred_executable_sql)
                break
            except Exception as e:
                pred_executable_sql += "    Wrong"
                print(pred_sql)
                print(e)
            
        
        final_sqls.append(pred_executable_sql)
    
    return final_sqls


def _test(opt):
    set_seed(opt.seed)
    print(opt)

    import time
    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    # initialize tokenizer
    if 'bart' in opt.save_path:
        tokenizer = BartTokenizerFast.from_pretrained(
            opt.save_path,
            add_prefix_space = True
        )
    else:
        tokenizer = T5TokenizerFast.from_pretrained(
            opt.save_path,
            add_prefix_space = True
        )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <"), AddedToken(" {"), AddedToken(" }")])

    if isinstance(tokenizer, BartTokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <"), AddedToken(" {"), AddedToken(" }")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    model_class = T5ForConditionalGeneration if "t5" in opt.save_path else BartForConditionalGeneration

    # initialize model
    model = model_class.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_pqls = []
    pql_ids = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 512,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            predict_pqls += decode_pqls(
                opt.db_path, 
                model_outputs, 
                batch_db_ids, 
                tokenizer,
                opt.dataset,
                opt.table_file_path
            )

            pql_ids += batch_db_ids
    
    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # predict_sqls是预测的结果
    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_pqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    
    if opt.mode == "eval":
        # initialize evaluator
        if opt.dataset == 'advising':
            evaluator = AdvisingEvaluator(opt.db_path)
        elif opt.dataset == 'restaurants':
            evaluator = RestaurantsEvaluator(opt.db_path)
        elif opt.dataset == 'wikisql':
            evaluator = WikiSQLEvaluator(opt.db_path, opt.table_file_path)
        elif opt.dataset == 'imdb':
            evaluator = IMDBEvaluator(opt.db_path)
        else:
            raise ValueError("Invalid dataset name")
        
        with open(opt.dev_filepath, "r") as f:
            data = json.load(f)
        
        ground_truth = []
        for d in data:
            ground_truth.append(d["output_sequence"].split("|")[-1].strip())
        
        em = evaluator.evaluate_logic_form(ground_truth, predict_pqls)
        exec = evaluator.evaluate_execution(ground_truth, predict_pqls, pql_ids)
        # exec = em
        print("EM:", em)
        print("EXEC:", exec)

        return em, exec
    


if __name__ == '__main__':
    # print("Read data...")
    # file = './output/advising/results_25143027'
    # data = read_data(file + '.txt')
    # # restaurants_evaluator = RestaurantsEvaluator('./data/restaurants-db.added-in-2020.sqlite')
    # advising_evaluator = AdvisingEvaluator('./data/advising-db.added-in-2020.sqlite')

    # print("Start evaluating...")
    # advising_evaluator.evaluate_execution(data[0], data[1], file)
    # # restaurants_evaluator.evaluate_execution(data[0], data[1])

    opt = parse_option()
    if opt.mode in ["eval", "test"]:
        _test(opt)

