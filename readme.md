[biaffine + ner](https://aclanthology.org/2020.acl-main.577.pdf)

### biaffine运算过程

```python

# -*- coding: utf8 -*-
#

import torch

# 假设768是mlp出来的hidden_size.
# batch_size, sequence_length, hidden_size = 32, 128,768

class Biaffine(object):
    def __init__(self, n_in=768, n_out=2, bias_x=True, bias_y=True):
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y

        self.weight = torch.rand(2, n_in + bias_x, n_in + bias_y)

    def forward(self, x, y):
        x = torch.cat((x, torch.ones_like(x[:, :, :1])), dim=-1) # torch.Size([32, 128, 769])
        y = torch.cat((y, torch.ones_like(y[:, :, :1])), dim=-1) # torch.Size([32, 128, 769])

        b = x.shape[0] # 32
        o = self.weight.shape[0] # 2

        x = x.unsqueeze(1).expand(-1, o, -1, -1) # torch.Size([32, 2, 128, 769])
        weight = self.weight.unsqueeze(0).expand(b, -1, -1, -1)  # torch.Size([32, 2, 769, 769])
        y = y.unsqueeze(1).expand(-1, o, -1, -1)  # torch.Size([32, 2, 128, 769])
        # torch.matmul(x, weight): torch.Size([32, 2, 128, 769])
        # y.permute((0, 1, 3, 2)).shape: torch.Size([32, 2, 769, 128])
        s = torch.matmul(torch.matmul(x, weight), y.permute((0, 1, 3, 2)))
        if s.shape[1] == 1:
            s = s.squeeze(dim=1)
        return s # torch.Size([32, 2, 128, 128])


if __name__ == '__main__':
    biaffine = Biaffine()
    x = torch.rand(32, 128, 768)
    y = torch.rand(32, 128, 768)
    print(biaffine.forward(x, y).shape)

```




### 运行

> 训练
```bash
python train.py
```

> 预测
```bash
python predict.py
```

## 结果

1. start_layer和end_layer都为linear的话，结果(当然有可能因为训练不充分,另外transformer也改了...)： 

```bash
2021-07-08 14:06:08,421 biaffine_ner.py [line:151] INFO Epoch 75, train loss: 0.0021, dev loss: 0.0209, dev precision: 0.8089, dev recall: 0.7991, dev f1:0.8022
```

2. 更改成专用mlp

```bash
2021-07-10 05:38:33,226 biaffine_ner.py [line:151] INFO Epoch 495, train loss: 0.0009, dev loss: 0.0086, dev precision: 0.8444, dev recall: 0.9355, dev f1:0.8826
2021-07-10 05:40:24,753 biaffine_ner.py [line:151] INFO Epoch 496, train loss: 0.0007, dev loss: 0.0086, dev precision: 0.8434, dev recall: 0.9355, dev f1:0.8820
2021-07-10 05:42:15,958 biaffine_ner.py [line:151] INFO Epoch 497, train loss: 0.0007, dev loss: 0.0086, dev precision: 0.8434, dev recall: 0.9353, dev f1:0.8820
2021-07-10 05:44:06,939 biaffine_ner.py [line:151] INFO Epoch 498, train loss: 0.0009, dev loss: 0.0086, dev precision: 0.8438, dev recall: 0.9358, dev f1:0.8824
2021-07-10 05:45:58,015 biaffine_ner.py [line:151] INFO Epoch 499, train loss: 0.0008, dev loss: 0.0086, dev precision: 0.8439, dev recall: 0.9358, dev f1:0.8825
2021-07-10 05:47:48,671 biaffine_ner.py [line:151] INFO Epoch 500, train loss: 0.0009, dev loss: 0.0086, dev precision: 0.8439, dev recall: 0.9358, dev f1:0.8825
```

3. 影响因素

其中sequence_length为128，如果超长的话，也是影响的一个点。
那么mlp那层可以设置n_out为500（比如说）。

在每一个batch里面，sentence长度基本一样，可以使用kmeans基于sentence length进行聚类，
然后再训练。


ner add mlp(done)
change linear to bidlinear. (new)
new dep(future.)


