import tqdm
import torch
import torch.nn as nn
from model import DKT
from torch.utils.data import DataLoader
from utils import read_data, DKTDataSet
from sklearn import metrics

MAX_STEPS = 50
NUM_OF_QUES = 1224
BATCH_SIZE = 64
LR = 0.001
HIDDEN_SIZE = 200
LAYERS = 1
EPOCHS = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
        # 论文公式(3)
        loss = torch.Tensor([0.0]).to(device)
        for student in range(pred.shape[0]):
            delta = batch[student][:, 0:NUM_OF_QUES] + batch[student][:,
                                                       NUM_OF_QUES:]  # [MAX_STEPS, NUM_OF_QUES] 获取练习的one_hot
            temp = pred[student][:MAX_STEPS - 1].mm(delta[1:].t())
            index = torch.LongTensor([[i for i in range(MAX_STEPS - 1)]]).to(device)
            p = temp.gather(0, index)[0]  # 只需要与下一时间步相乘的数据
            a = (((batch[student][:, 0:NUM_OF_QUES] - batch[student][:, NUM_OF_QUES:]).sum(1) + 1) // 2)[1:]
            for i in range(len(p)):
                if p[i] > 0:  # 补的数据one_hot全为0，所以最终计算结果必为0
                    loss = loss - (a[i] * torch.log(p[i]) + (1 - a[i]) * torch.log(1 - p[i]))
        return loss


if __name__ == '__main__':

    train_ques, train_ans = read_data('./data/statics2011/static2011_train.txt', MAX_STEPS)
    test_ques, test_ans = read_data('./data/statics2011/static2011_test.txt', MAX_STEPS)

    train_set = DKTDataSet(train_ques, train_ans, MAX_STEPS, NUM_OF_QUES)
    test_set = DKTDataSet(test_ques, test_ans, MAX_STEPS, NUM_OF_QUES)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = DKT(2 * NUM_OF_QUES, HIDDEN_SIZE, LAYERS, NUM_OF_QUES).to(device)

    loss_func = lossFunc()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        # ========================================Train=============================================================

        for data in tqdm.tqdm(train_loader):
            data = data.to(device)
            pred = model(data)
            loss = loss_func(pred, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ========================================Test=============================================================

        ground_truth = []
        prediction = []
        for data in tqdm.tqdm(test_loader):
            data = data.to(device)
            pred = model(data).cpu()
            data = data.cpu()
            for student in range(pred.shape[0]):
                delta = data[student][:, 0:NUM_OF_QUES] + data[student][:, NUM_OF_QUES:]
                temp = pred[student][:MAX_STEPS - 1].mm(delta[1:].t())
                index = torch.LongTensor([[i for i in range(MAX_STEPS - 1)]])
                p = temp.gather(0, index)[0]
                a = (((data[student][:, 0:NUM_OF_QUES] - data[student][:, NUM_OF_QUES:]).sum(1) + 1) // 2)[1:]
                for i in range(len(p)):
                    if p[i] > 0:
                        ground_truth.append(a[i].item())
                        prediction.append(p[i].item())

        fpr, tpr, thresholds = metrics.roc_curve(ground_truth, prediction)
        auc = metrics.auc(fpr, tpr)

        print('Epoch {:02d} | AUC {:.4f}'.format(epoch, auc))