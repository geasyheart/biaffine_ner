# -*- coding: utf8 -*-
#
import torch


class Metrics(object):
    def __init__(self):
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.steps = 0

    def step(self, y_true, y_pred):
        y_pred = y_pred.argmax(axis=-1)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        correct = torch.eq(y_true, y_pred)

        # escape zero.
        ones = torch.ones_like(y_pred)
        zeros = torch.zeros_like(y_pred)
        y_pred = torch.where(y_pred < 1, zeros, ones)

        ones = torch.ones_like(y_true)
        zeros = torch.zeros_like(y_true)
        y_true = torch.where(y_true < 1, zeros, ones)

        correct = torch.mul(correct, y_true)

        precision = torch.sum(correct) / (torch.sum(y_pred) + 1e-8)
        recall = torch.sum(correct) / (torch.sum(y_true) + 1e-8)
        f1 = 2 * recall * precision / (precision + recall + 1e-8)

        self.precision += precision
        self.recall += recall
        self.f1 += f1
        self.steps += 1

        return precision, recall, f1

    def summary(self):
        return self.precision, self.recall, self.f1
