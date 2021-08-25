### Deep Knowledge Tracing

参考：https://chsong.live/20201124_DKT-Pytorch/index.html



**utils.py**

read_data()：读取数据，对于长度不足的数据需要进行padding

DKTDataSet：重写Dataset类，当给定题目和答案时，返回此次交互的one_hot形式



**model.py**

LSTM + Linear



**main.py**

lossFunc：实现论文中的损失函数：$L = \sum_t\mathcal{l} (\textbf{y}_i^T \delta (q_{t+1}), a_{t+1})$
