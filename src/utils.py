# -*- coding: utf8 -*-
#
import json
import os
import sys
from typing import Dict, Iterable, List

import torch
from torch.utils.data import dataset, dataloader
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
import logging


def get_logger():
    logger = logging.getLogger('biaffine')
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'
    file_handler = logging.FileHandler(filename='run.log')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(logging.Formatter(fmt))
    stream_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = get_logger()

DATA_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )),
    'data',
)

TRAIN_FILE = os.path.join(DATA_PATH, "msra_ner", "train.tsv")
TEST_FILE = os.path.join(DATA_PATH,  "msra_ner", "dev.tsv")


# def read_data(file: str) -> Iterable[Dict]:
#     with open(file, encoding='utf-8') as f:
#         for line in f:
#             line = json.loads(line)
#             if line['entity_list']:
#                 yield line

#
# def get_labels(label_file: str = os.path.join(DATA_PATH, 'label_map.json')) -> Dict[str, int]:
#     if os.path.exists(label_file):
#         with open(label_file, encoding='utf-8') as f:
#             return json.loads(f.read())
#
#     label_maps = {}
#     for line in read_data(TRAIN_FILE):
#         for entity in line['entity_list']:
#             entity_type = entity['type']
#             value = label_maps.get(entity_type)
#             if value is not None:
#                 pass
#             else:
#                 label_maps[entity_type] = len(label_maps) + 1
#     with open(label_file, 'w', encoding='utf-8') as f:
#         f.write(json.dumps(
#             label_maps,
#             ensure_ascii=False,
#             indent=2
#         ))
#     return label_maps
#
#
# def get_start_index(indices1: List[int], indices2: List[int]):
#     lens = len(indices1)
#     for i in range(len(indices2)):
#         if indices2[i: i + lens] == indices1:
#             return i
#
#
# class MyDataSet(dataset.Dataset):
#     def __init__(self, file: str, transformer: str, batch_size: int = 32, shuffle=True, max_length: int = 128,
#                  device=None):
#         self.file = file
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.max_length = max_length
#         self.device = device
#
#         self.tokenizer = AutoTokenizer.from_pretrained(transformer)
#         self.label_map = get_labels()
#         self.datas = [_ for _ in read_data(file=file)]
#
#     def __getitem__(self, index) -> T_co:
#         line = self.datas[index]
#         text, entity_list = line['text'], line['entity_list']
#         text_embed = self.tokenizer.encode_plus(text=text, padding="max_length", max_length=self.max_length, truncation=True)
#
#         mask = [text_embed['attention_mask'] for i in range(sum(text_embed['attention_mask']))]
#         line_zeros = [0] * self.max_length
#         mask.extend([line_zeros for i in range(len(mask), self.max_length)])
#
#         label_mask = torch.zeros(self.max_length, self.max_length, dtype=torch.long)
#
#         for entity in entity_list:
#             entity_type = entity['type']
#             entity_argument = entity['argument']
#             entity_type_id = self.label_map.get(entity_type, 0)
#
#             entity_input_ids = self.tokenizer.encode_plus(entity_argument, add_special_tokens=False)['input_ids']
#             start_index = get_start_index(indices1=entity_input_ids, indices2=text_embed['input_ids'])
#             # truncated
#             if start_index is None:
#                 continue
#             end_index = start_index + len(entity_input_ids)
#             label_mask[start_index, end_index] = entity_type_id
#         return (
#             torch.tensor(text_embed['input_ids'], dtype=torch.long, device=self.device),
#             torch.tensor(text_embed['token_type_ids'], dtype=torch.long, device=self.device),
#             torch.tensor(text_embed['attention_mask'], dtype=torch.long, device=self.device),
#             torch.tensor(mask, dtype=torch.long, device=self.device),
#             label_mask.to(self.device)
#         )
#
#     def __len__(self):
#         return len(self.datas)
#
#     def to_dataloader(self):
#         return dataloader.DataLoader(dataset=self, batch_size=self.batch_size, shuffle=self.shuffle)


if __name__ == '__main__':
    for data in MyDataSet(TEST_FILE, 'ckiplab/albert-tiny-chinese').to_dataloader():
        print(data)
