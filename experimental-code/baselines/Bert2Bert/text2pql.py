import argparse
import os
import torch
import transformers
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed

from utils.load_dataset import Text2PQLDataset
from utils.evaluate import evaluate_logic_form
from utils.plot import plot_loss

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
    parser.add_argument('--model_name_or_path', type = str, default = "t5-3b",
                        help = 
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help = 'whether to use adafactor optimizer.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')
    parser.add_argument('--train_filepath', type = str, default = "data/advising/advising_train.json",
                        help = 'file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type = str, default = "data/advising/advising_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')
    parser.add_argument("--output", type = str, default = "predicted_pql.txt",
                help = "save file of the predicted sqls.")
    
    opt = parser.parse_args()

    return opt

def _train(opt):
    set_seed(opt.seed)
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    tokenizer = BertTokenizer.from_pretrained(opt.model_name_or_path, add_prefix_space = True)

    train_dataset = Text2PQLDataset(opt.train_filepath, opt.mode)

    train_dataloader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = True, collate_fn = lambda x: x)

    encoder = BertGenerationEncoder.from_pretrained(opt.model_name_or_path)
    decoder = BertGenerationDecoder.from_pretrained(opt.model_name_or_path, add_cross_attention = True, is_decoder = True)
    model = EncoderDecoderModel(encoder = encoder, decoder = decoder)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if torch.cuda.is_available():
        model.cuda()

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
    # save checkpoint for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider's training set)
    num_checkpoint_steps = int(5 * 1.42857 * len(train_dataset)/opt.batch_size)

    optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    plot_loss_values = []

    model.train()
    train_step = 0
    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch+1}.")
        for batch in train_dataloader:
            train_step += 1
            # print(batch)
            
            batch_inputs = [data[0] for data in batch]
            batch_pqls = [data[1] for data in batch]
            
            if epoch == 0:
                for batch_id in range(len(batch_inputs)):
                    print(batch_inputs[batch_id])
                    print(batch_pqls[batch_id])
                    print("----------------------")

            tokenized_inputs = tokenizer(
                batch_inputs, 
                padding = "max_length",
                return_tensors = "pt",
                max_length = 512,
                truncation = True,
            )
            
            tokenized_outputs = tokenizer(
                batch_pqls, 
                padding = "max_length", 
                return_tensors = 'pt',
                max_length = 512,
                truncation = True
            )
            
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]
            labels = tokenized_outputs["input_ids"]
            decoder_attention_mask = tokenized_outputs["attention_mask"]
            labels[labels == tokenizer.pad_token_id] = -100
            

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                decoder_attention_mask = decoder_attention_mask.cuda()
            
            loss = model(
                input_ids=input_ids, 
                labels=labels, 
                attention_mask=attention_mask, 
                decoder_attention_mask=decoder_attention_mask, 
                return_dict=True).loss
            loss.backward()

            plot_loss_values.append(loss.item())

            if scheduler is not None:
                scheduler.step()

            if train_step % opt.gradient_descent_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if train_step % num_checkpoint_steps == 0 and epoch >= 6:
                print(f"At {train_step} training step, save a checkpoint.")
                os.makedirs(opt.save_path, exist_ok = True)
                model.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))
                tokenizer.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))

    print(f"At {train_step} training step, save a checkpoint.")
    os.makedirs(opt.save_path, exist_ok = True)
    model.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))
    tokenizer.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))

    plot_loss(plot_loss_values)

def _test(opt):
    set_seed(opt.seed)
    print(opt)

    import time
    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    tokenizer = BertTokenizer.from_pretrained(opt.save_path, add_prefix_space = True)

    dev_dataset = Text2PQLDataset(opt.dev_filepath, opt.mode)

    dev_dataloader = DataLoader(dev_dataset, batch_size = opt.batch_size, shuffle = True, collate_fn = lambda x: x)

    encoder = BertGenerationEncoder.from_pretrained(opt.save_path)
    decoder = BertGenerationDecoder.from_pretrained(opt.save_path, add_cross_attention = True, is_decoder = True)
    model = EncoderDecoderModel(encoder = encoder, decoder = decoder)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if torch.cuda.is_available():
        model.cuda()


    model.eval()
    predicted_pqls = []
    for batch in tqdm(dev_dataloader):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True,
            add_special_tokens = False
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = attention_mask,
                max_length = 512,
            )

            # print(model_outputs.shape)

            for batch_id in range(model_outputs.shape[0]):
                pred_sequence = tokenizer.decode(model_outputs[batch_id], skip_special_tokens = True)
                predicted_pqls.append(pred_sequence)

    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)

    with open(opt.output, 'w', encoding = 'utf-8') as f:
        for pql in predicted_pqls:
            f.write(pql + "\n")

    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))

    with open(opt.dev_filepath, "r") as f:
            data = json.load(f)
        
    ground_truth = []
    for d in data:
        ground_truth.append(d["output_sequence"].split("|")[-1].strip())

    em = evaluate_logic_form(ground_truth, predicted_pqls)
    exec = em
    print('EM: ', em)
    print('EXEC: ', exec)
    return em, exec


if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        _test(opt)