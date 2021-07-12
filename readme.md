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


