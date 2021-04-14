import torch
import torch.nn as nn
import torch.utils.data as Data
import argparse
import time
from tqdm import tqdm
from model import DeepFM
from utils import create_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def cal_acc(outputs, labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy


def training(model, train_loader, valid_loader, batch_size, lr, epochs, device):
    loss = nn.BCELoss()
    loss = loss.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    best_auc = 0

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        total_loss, total_acc = 0, 0
        for i, x in enumerate(train_loader):
            sparse_data, dense_data, labels = x[0], x[1], x[2]
            sparse_data, dense_data, labels = sparse_data.to(device), dense_data.to(device), labels.to(device)
            outputs = model((sparse_data, dense_data)).view(-1)
            batch_loss = loss(outputs, labels)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_acc += cal_acc(outputs, labels) / batch_size
            if (i+1) % 100 == 0 or (i + 1) == len(train_loader):
                print('Epoch {:02d} | Step {:04d} / {} | ACC:{:.3f} | Loss {:.4f} | Time {:.4f}'.format(epoch, i+1, len(
                    train_loader), total_acc / (i + 1) * 100, total_loss / (i + 1), time.time() - start_time))
        scheduler.step()

        model.eval()
        start_time = time.time()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            valid_preds, valid_labels = [], []
            for i, x in enumerate(tqdm(valid_loader, desc='Valid')):
                sparse_data, dense_data, labels = x[0], x[1], x[2]
                sparse_data, dense_data, labels = sparse_data.to(device), dense_data.to(device), labels.to(device)

                outputs = model((sparse_data, dense_data)).view(-1)
                valid_preds.extend(outputs.cpu().numpy().tolist())
                valid_labels.extend(labels.cpu())

                total_loss += loss(outputs, labels).item()
                total_acc += cal_acc(outputs, labels) / batch_size

        cur_auc = roc_auc_score(valid_labels, valid_preds)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model, 'ckpt.model')
        print('Epoch {:02d} | Valid | AUC:{:.4f} | BestAUC:{:.4f} |ACC:{:.3f} | Loss {:.4f} | Time {:.4f}'.format(
            epoch, cur_auc, best_auc, total_acc / (len(valid_loader)) * 100, total_loss / len(valid_loader), time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFM')
    parser.add_argument('--test_size', type=int, default=0.2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--file', type=str, default='criteo_sampled_data.csv')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # =============================================Data=================================================================
    data, feat_columns, dense_feats, sparse_feats = create_dataset(file=args.file)
    train, valid = train_test_split(data, test_size=args.test_size)
    train_dataset = Data.TensorDataset(torch.LongTensor(train[sparse_feats].values),
                                       torch.FloatTensor(train[dense_feats].values),
                                       torch.FloatTensor(train['label'].values))
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = Data.TensorDataset(torch.LongTensor(valid[sparse_feats].values),
                                       torch.FloatTensor(valid[dense_feats].values),
                                       torch.FloatTensor(valid['label'].values))
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)
    # =============================================Model================================================================
    model = DeepFM(feat_columns, args.embed_dim)
    model.to(device)
    # =============================================Train================================================================
    training(model, train_loader, valid_loader, args.batch_size, args.lr, args.epochs, device)

