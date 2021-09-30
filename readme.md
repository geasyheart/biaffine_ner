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

        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))


    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        
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



