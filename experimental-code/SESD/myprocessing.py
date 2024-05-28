import json
import random


def remove_sketelon(path, new_path):
    with open(path, 'r') as f:
        data = json.load(f)
    new_data = []
    for d in data:
        # input_sequence = d['input_sequence']
        tmp = d['input_sequence'].split('|')
        input_sequence = tmp[0].strip() + ' | ' + tmp[2].strip()
        # input_sequence = tmp[0].strip() + ' |' + '|'.join(tmp[3:])
        # output_sequence = d['output_sequence']
        output_sequence = d['output_sequence'].split('|')
        # output_sequence[0] = ' '
        # output_sequence = '|'.join(output_sequence)
        output_sequence = output_sequence[1].strip()
        new_data.append({'input_sequence': input_sequence, 'output_sequence': output_sequence, 'db_id': d['db_id'], 'tc_original': d['tc_original']})
    
    with open(new_path, 'w') as f:
        json.dump(new_data, f, indent=2)


def split_advising():
    with open('./data/advising/advising_preprocessed.json', 'r') as f:
        data = json.load(f)
    
    print(len(data))
    train_data = []
    dev_data = []
    test_data = []
    # random.seed(42)
    # shuffle data
    random.shuffle(data)

    for d in data:
        if random.random() < 0.1:
            dev_data.append(d)
        elif random.random() < 0.2:
            test_data.append(d)
        else:
            train_data.append(d)
    
    print(len(train_data), len(dev_data), len(test_data))
    
    with open('./data/advising/advising_train.json', 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open('./data/advising/advising_dev.json', 'w') as f:
        json.dump(dev_data, f, indent=2, ensure_ascii=False)
    with open('./data/advising/advising_test.json', 'w') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    remove_sketelon('data/wikisql/wikisql_train_preprocessed.json', 'data/wikisql/t5/wikisql_train_preprocessed.json')
    remove_sketelon('data/wikisql/wikisql_dev_preprocessed.json', 'data/wikisql/t5/wikisql_dev_preprocessed.json')
    remove_sketelon('data/wikisql/wikisql_test_preprocessed.json', 'data/wikisql/t5/wikisql_test_preprocessed.json')
    # remove_sketelon('data/imdb/cross_validation/split_4/test.json', 'data/imdb/cross_validation_t5/split_4/test.json')
    # remove_sketelon('data/imdb/cross_validation/split_4/train.json', 'data/imdb/cross_validation_t5/split_4/train.json')
    # remove_sketelon('data/advising/advising_train.json', 'data/advising/advising_train_t5.json')
    # remove_sketelon('data/advising/advising_test.json', 'data/advising/advising_test_t5.json')
    # remove_sketelon('data/advising/advising_dev.json', 'data/advising/advising_dev_t5.json')
    # # remove_sketelon('data/restaurants/cross_validation/split_4/train.json', 'data/restaurants/t5/split_4/train.json')
    # remove_sketelon('data/restaurants/cross_validation/split_4/test.json', 'data/restaurants/t5/split_4/test.json')
    # split_advising()