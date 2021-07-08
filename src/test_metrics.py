# -*- coding: utf8 -*-
# 计算指标可参考：https://zhuanlan.zhihu.com/p/64315175
import torch

from src.model.metrics import cal_metrics
from sklearn.metrics import classification_report, f1_score


def version1():
    p_count = [0, 0, 0]
    r_count = [0, 0, 0]
    t_count = [0, 0, 0]

    for p, t in zip(predicts, trues):
        p_count[p - 1] += 1
        t_count[t - 1] += 1
        if p == t:
            r_count[p - 1] += 1

    p_s = [r / p for r, p in zip(r_count, p_count)]
    r_s = [r / t for r, t in zip(r_count, t_count)]
    p_ = sum(p_s) / 3
    r_ = sum(r_s) / 3
    print(p_, r_)


predicts = torch.tensor([1, 1, 2, 3, 2, 2, 3, 2, 3])
trues = torch.tensor([1, 1, 1, 1, 2, 2, 2, 3, 3])
# version1()

print(cal_metrics(torch.tensor(predicts), torch.tensor(trues)))
print(classification_report(trues, predicts))
print(f1_score(trues, predicts, average='macro'))
"""
              precision    recall  f1-score   support

           1       1.00      0.50      0.67         4
           2       0.50      0.67      0.57         3
           3       0.33      0.50      0.40         2

    accuracy                           0.56         9
   macro avg       0.61      0.56      0.55         9
weighted avg       0.69      0.56      0.58         9

[0.66666667 0.57142857 0.4       ]

"""