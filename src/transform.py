# -*- coding: utf8 -*-
#
import json
import os
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataset, dataloader
from transformers import AutoTokenizer

from src.utils import DATA_PATH, TRAIN_FILE


def read_tsv_as_sentence(file_path, delimiter='\t'):
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            cells = line.split(delimiter)
            if line and cells:
                sentence.append(cells)
            else:
                if sentence:
                    yield sentence
                    sentence = []

    if sentence:
        yield sentence


def get_start_and_end_indices(labels: List[str]):
    def _get_end_index(start_index):
        c = 0
        for ca in labels[start_index+1:]:
            if ca.startswith('M') or ca.startswith('E'):
                c += 1
            else:
                break
        return start_index + c

    indices = []
    label_map = get_labels()
    for index, char in enumerate(labels):
        if char.startswith('B'):
            label = label_map[char]
            end_index = _get_end_index(start_index=index)
            indices.append(((index, end_index), label))
    return indices


def get_labels(label_path=os.path.join(DATA_PATH, 'msra_ner', 'label_map.json')):
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            return json.loads(f.read())
    lines = read_tsv_as_sentence(TRAIN_FILE)

    label_map = {'[PAD]': 0}
    for line in lines:
        line = [i[0].split(' ') for i in line]
        labels = [i[1] for i in line]
        for label in labels:
            label_map.setdefault(label, len(label_map))
    with open(label_path, 'w') as f:
        f.write(json.dumps(label_map))
    return label_map


class Transform1DataSet(dataset.Dataset):
    def __init__(self, file: str, transformer: str, batch_size: int = 32, shuffle=True, max_length: int = 128,
                 device=None):
        self.file = file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(transformer) if isinstance(transformer, str) else transformer
        self.labels = get_labels()
        self.lines = list(read_tsv_as_sentence(file))

    def __getitem__(self, item):
        line = [i[0].split(' ') for i in self.lines[item]]
        tokens = [i[0] for i in line]
        labels = [i[1] for i in line]

        input_ids = [self.tokenizer.cls_token_id] + \
                     self.tokenizer.convert_tokens_to_ids(tokens)[:self.max_length - 2] + \
                    [self.tokenizer.sep_token_id]
        input_ids = torch.tensor(input_ids + (self.max_length - len(input_ids)) * [self.tokenizer.pad_token_id], dtype=torch.long)

        label_id_matrix = torch.zeros(self.max_length, self.max_length, dtype=torch.long)
        label_indices = get_start_and_end_indices(labels=labels)
        for (start_index, end_index), label_id in label_indices:
            try:
                label_id_matrix[start_index, end_index] = label_id
            except IndexError:
                # ignore 
                pass

        mask = torch.zeros(self.max_length, self.max_length)
        mask[:len(tokens), :len(tokens)] = 1
        return input_ids.to(self.device), label_id_matrix.to(self.device), mask.triu().bool().to(self.device)

    def __len__(self):
        return len(self.lines)

    def to_dataloader(self):
        return dataloader.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=self.shuffle)


if __name__ == '__main__':
    for data in Transform1DataSet(
            file=TRAIN_FILE,
            transformer='hfl/chinese-electra-180g-small-discriminator'
    ).to_dataloader():
        print(data)
