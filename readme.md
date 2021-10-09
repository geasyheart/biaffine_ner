[biaffine + ner](https://aclanthology.org/2020.acl-main.577.pdf)

### 更新

在msra数据集上最终结果跑出来最低类别f1 能达到0.8+，这个可以根据这个代码直接跑出来就行，具体优化不进行了。


预测出来的结果如下：
```md
真实标签： tensor([[ 0, 10, 11]], device='cuda:0') 
预测标签： tensor([[ 0, 10, 11]], device='cuda:0')

真实标签： tensor([[ 0, 84, 85],
        [ 0, 87, 88]], device='cuda:0') 
预测标签： tensor([[ 0, 84, 71],
        [ 0, 84, 81],
        [ 0, 84, 85],
        [ 0, 87, 84],
        [ 0, 87, 85],
        [ 0, 87, 88],
        [ 0, 88, 85]], device='cuda:0')

真实标签： tensor([[0, 0, 5]], device='cuda:0') 
预测标签： tensor([[ 0,  0,  5],
        [ 0, 42,  5]], device='cuda:0')

真实标签： tensor([[ 0, 17, 18],
        [ 0, 20, 22]], device='cuda:0') 
预测标签： tensor([[ 0, 17, 18],
        [ 0, 20, 18],
        [ 0, 20, 22]], device='cuda:0')

真实标签： tensor([[0, 3, 8]], device='cuda:0') 
预测标签： tensor([[0, 3, 8]], device='cuda:0')

真实标签： tensor([[ 0, 13, 18],
        [ 0, 22, 24],
        [ 0, 36, 41]], device='cuda:0') 
预测标签： tensor([[ 0, 13, 18],
        [ 0, 22, 18],
        [ 0, 22, 24],
        [ 0, 36, 18],
        [ 0, 36, 22],
        [ 0, 36, 24],
        [ 0, 36, 41],
        [ 0, 41, 18],
        [ 0, 41, 24]], device='cuda:0')

```

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



