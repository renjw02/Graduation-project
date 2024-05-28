import argparse
import json
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/restaurants', help='Path to the data folder')
    parser.add_argument('--output_path', type=str, default='output', help='Path to the output folder')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    return parser.parse_args()

def split_dataset(opt):
    data_path = opt.data_path
    output_path = opt.output_path
    n_splits = opt.n_splits
    seed = opt.seed
    random.seed(seed)

    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)

    random.shuffle(data)
    data_len = len(data)
    split_size = data_len // n_splits
    splits = [data[i:i+split_size] for i in range(0, data_len, split_size)]
    
    for i in range(n_splits):
        train_data = []
        test_data = splits[i]
        for j in range(n_splits):
            if j != i:
                train_data.extend(splits[j])
        
        # Save the split data
        split_dir = os.path.join(output_path, f'split_{i}')
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, 'train.json'), 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(os.path.join(split_dir, 'test.json'), 'w') as f:
            json.dump(test_data, f, indent=2)





if __name__ == '__main__':
    opt = parse_args()
    split_dataset(opt)