# -*- coding: utf8 -*-

#

from torch import nn

from src.model.dropout import SharedDropout


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, activation=True):
        super(MLP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = SharedDropout(p=dropout)
        self.reset_parameters()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
