# -*- coding: utf8 -*-
#
from torch import nn
from transformers import AutoModel

from src.model.biaffine import Biaffine
from src.model.mlp import MLP


class BiaffineNerModel(nn.Module):

    def __init__(self, transformer: str, sequence_length: int, n_labels: int):
        super(BiaffineNerModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(transformer)

        self.start_layer = MLP(n_in=self.encoder.config.hidden_size, n_out=sequence_length, dropout=0.33)
        self.end_layer = MLP(n_in=self.encoder.config.hidden_size, n_out=sequence_length, dropout=0.33)

        self.biaffine = Biaffine(n_in=sequence_length, n_out=n_labels, bias_x=True, bias_y=True)

    def forward(self, input_ids):
        bert_out = self.encoder(input_ids=input_ids)
        start_embed = self.start_layer(bert_out[0])
        end_embed = self.end_layer(bert_out[0])

        biaffine_out = self.biaffine(x=start_embed, y=end_embed)
        # [batch_size, seq_len, seq_len, n_labels]
        result = biaffine_out.permute(0, 2, 3, 1)
        return result.contiguous()
