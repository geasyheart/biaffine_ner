# -*- coding: utf8 -*-

#

import torch
from torch import nn


class SharedDropout(nn.Module):
    """
        SharedDropout differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.
        Args:
            p (float):
                The probability of an element to be zeroed. Default: 0.5.
            batch_first (bool):
                If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
                Default: ``True``.
        Examples:
            >>> x = torch.ones(1, 3, 5)
            >>> nn.Dropout()(x)
            tensor([[[0., 2., 2., 0., 0.],
                     [2., 2., 0., 2., 2.],
                     [2., 2., 2., 2., 0.]]])
            >>> SharedDropout()(x)
            tensor([[[2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.]]])

        意思就是说：
        在y和z轴乘以相同的数值

        假设 mask 为:
        [0, 2, 2, 0, 0]

        y/z:
        [
            [1, 1,1,1,1],
            [1, 1,1,1,1],
        ]
        就变成了:

        [
            [0, 2,2,0,0],
            [0, 2,2,0, 0],
        ]

        """

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        """

        :param x:
        :return:
        """

        if not self.training:
            return x
        if self.batch_first:
            mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
        else:
            mask = self.get_mask(x[0], self.p)
        x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)
