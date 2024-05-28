from __future__ import unicode_literals, print_function, division

from collections import defaultdict
from io import open
import unicodedata
import torch
from constant import MAX_LENGTH, SOS_token, EOS_token
import nltk
import pandas as pd
import re
import json

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class LanguageUtil:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        # for word in sentence:
        #     self.addWord(word)
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, file_path):
    print(f"Reading file {file_path} ...")
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as lines:
        for line in lines:
            info = json.loads(line)
            en = info['question']
            query = info['query']
            if info.get('table_id'):
                table_id = info['table_id']
                pairs.append([en, query, table_id])
            else:
                pairs.append([en, query])
    # Read the file and split into lines
    # lines = open('data_/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
    #     read().strip().split('\n')

    # # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # print(pairs[:5])

    # pairs = [list(reversed(p)) for p in pairs]
    input_lang = LanguageUtil(lang1)
    output_lang = LanguageUtil(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, split='train'):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, split)
    print("Read %s sentence pairs" % len(pairs))
    # pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


