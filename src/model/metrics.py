# -*- coding: utf8 -*-
#
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from umetrics import MacroMetrics
from src.transform import get_labels

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
        self.metrics = MacroMetrics(labels=ALL_LABELS)

    def step(self, y_true, y_pred, mask):
        """version2"""
        mask = mask.view(-1).to('cpu')

        y_pred = y_pred.argmax(axis=-1)
        y_trues = y_true.view(-1)
        y_preds = y_pred.view(-1)

        y_trues = y_trues.to('cpu') * mask
        y_preds = y_preds.to('cpu') * mask

        self.metrics.step(y_trues=y_trues.tolist(), y_preds=y_preds.tolist())

    def summary(self):
        self.metrics.classification_report()
        return self.metrics.precision_score(), self.metrics.recall_score(), self.metrics.f1_score()
