biaffine + ner.

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

1. start_layer和end_layer都为linear的话，结果： 

```bash
2021-07-08 14:06:08,421 biaffine_ner.py [line:151] INFO Epoch 75, train loss: 0.0021, dev loss: 0.0209, dev precision: 0.8089, dev recall: 0.7991, dev f1:0.8022
```

2. 更改成专用mlp

```bash

```

3. 影响因素

其中sequence_length为128，如果超长的话，也是影响的一个点。
那么mlp那层可以设置n_out为500（比如说）。

在每一个batch里面，sentence长度基本一样，可以使用kmeans基于sentence length进行聚类，
然后再训练。




