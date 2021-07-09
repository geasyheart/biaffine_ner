# -*- coding: utf8 -*-
#
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils import get_labels

ALL_LABELS = list(get_labels().values())


def cal_metrics(y_preds, y_trues):
    """

    :param y_preds:
    :param y_trues:
    :return:
    """
    y_preds_unique_labels = torch.unique(y_preds)
    y_trues_unique_labels = torch.unique(y_trues)

    all_labels = torch.cat((y_preds_unique_labels, y_trues_unique_labels)).unique(sorted=True)
    # ignore 0
    if 0 in all_labels:
        all_labels = all_labels[1:]

    y_preds_labels, y_preds_count = y_preds.unique(return_counts=True)
    y_trues_labels, y_trues_count = y_trues.unique(return_counts=True)

    corrects_mask = torch.eq(y_preds, y_trues)
    corrects_labels, corrects_count = y_trues[corrects_mask].unique(return_counts=True)

    y_preds_map = dict(zip(y_preds_labels.tolist(), y_preds_count.tolist()))
    y_true_map = dict(zip(y_trues_labels.tolist(), y_trues_count.tolist()))
    corrects_map = dict(zip(corrects_labels.tolist(), corrects_count.tolist()))
    precision, recall, f1 = 0, 0, 0
    for label in all_labels.tolist():
        _precision = corrects_map.get(label, 0) / (y_preds_map.get(label, 0) + 1e-8)
        _recall = corrects_map.get(label, 0) / (y_true_map.get(label, 0) + 1e-8)
        _f1 = 2 * _precision * _recall / (_precision + _recall + 1e-8)
        precision += _precision
        recall += _recall
        f1 += _f1

    all_label_count = len(all_labels)
    return precision / all_label_count, recall / all_label_count, f1 / all_label_count


class Metrics(object):
    def __init__(self):
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.steps = 0

    def old(self, y_true, y_pred):
        """old."""
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

    def version1(self, y_true, y_pred):
        self.steps += 1

        y_pred = y_pred.argmax(axis=-1)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        escape_0_y_true = y_true[y_true.not_equal(0)]
        if escape_0_y_true.size(0) == 0:
            return 0., 0., 0.
        escape_0_y_pred = y_pred[y_pred.not_equal(0)]
        if escape_0_y_pred.size(0) == 0:
            return 0., 0., 0.

        y_true_min, y_true_max = escape_0_y_true.min(), escape_0_y_true.max()

        rights = [0 for i in torch.arange(y_true_min, y_true_max + 1)]
        predicts = [0 for i in torch.arange(y_true_min, y_true_max + 1)]
        totals = [0 for i in torch.arange(y_true_min, y_true_max + 1)]
        for y1, y2 in zip(y_true, y_pred):
            if y1 != 0:
                totals[y1 - 1] += 1
            if y2 != 0:
                predicts[y2 - 1] += 1
            if y1 == y2 and y1 != 0:
                rights[y1 - 1] += 1

        precision = sum([c / (p + 1e-8) for c, p in zip(rights, predicts)]) / (len(predicts) + 1e-8)
        recall = sum([c / (t + 1e-8) for c, t in zip(rights, totals)]) / (len(totals) + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        self.precision += precision
        self.recall += recall
        self.f1 += f1

        return precision, recall, f1

    def step(self, y_true, y_pred):
        """version2"""
        self.steps += 1

        y_pred = y_pred.argmax(axis=-1)
        y_trues = y_true.view(-1)
        y_preds = y_pred.view(-1)

        # precision, recall, f1 = cal_metrics(y_preds=y_preds, y_trues=y_trues)
        y_trues = y_trues.to('cpu')
        y_preds = y_preds.to('cpu')
        precision = precision_score(y_trues, y_preds, average='macro', zero_division=0, labels=ALL_LABELS)
        recall = recall_score(y_trues, y_preds, average='macro', zero_division=0, labels=ALL_LABELS)
        f1 = f1_score(y_trues, y_preds, average='macro', zero_division=0, labels=ALL_LABELS)
        self.precision += precision
        self.recall += recall
        self.f1 += f1

        return precision, recall, f1

    def summary(self):
        return self.precision / self.steps, self.recall / self.steps, self.f1 / self.steps
