from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from LanguageUtil import *
from constant import *
import random

def get_dataloader(batch_size = 32, file_path='data/train_paired.jsonl'):
    input_lang, output_lang, pairs = prepareData('eng', 'query', file_path)

    n = len(pairs)
    train_size = int(0.8 * n)
    test_size = n - train_size
    random.shuffle(pairs)
    train_pairs = pairs[:train_size]
    test_pairs = pairs[train_size:]
    

    input_ids = np.zeros((train_size, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((train_size, MAX_LENGTH), dtype=np.int32)

    # 80% of the data is used for training, 20% for testing
    if len(train_pairs[0]) == 3:
        for idx, (inp, tgt, _) in enumerate(train_pairs):
            inp_ids = indexesFromSentence(input_lang, inp)
            tgt_ids = indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids
    elif len(train_pairs[0]) == 2:
        for idx, (inp, tgt) in enumerate(train_pairs):
            inp_ids = indexesFromSentence(input_lang, inp)
            tgt_ids = indexesFromSentence(output_lang, tgt)
            inp_ids.append(EOS_token)
            tgt_ids.append(EOS_token)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader, test_pairs


def plot_data(x, y, xlabel = "x", ylabel = "y", label = 'plot'):
	plt.figure()
	plt.plot(x, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	print("Generating plot for ", label)
	plt.savefig("./loss" + label + ".png")