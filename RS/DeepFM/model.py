import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, feat_columns, emb_size):
        super().__init__()

        dense_feats, sparse_feats = feat_columns[0], feat_columns[1]
        self.dense_size = len(dense_feats)
        self.sparse_size = len(sparse_feats) * emb_size

        '''FM'''
        self.w = nn.Linear(self.dense_size, 1, bias=True)
        self.sparse_first_emd = nn.ModuleList([nn.Embedding(feat['feat_num'], 1) for feat in sparse_feats])
        self.sparse_second_emd = nn.ModuleList([nn.Embedding(feat['feat_num'], emb_size) for feat in sparse_feats])

        '''DNN'''
        self.dnn = nn.Sequential(
            nn.Linear(self.dense_size + self.sparse_size, 200),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        sparse_inputs, dense_inputs = inputs

        '''FM一阶'''
        fm_first_dense = self.w(dense_inputs)                                 # [batch_size, 1]
        fm_first_sparse = torch.cat([self.sparse_first_emd[i](sparse_inputs[:, i])for i in range(sparse_inputs.shape[1])], -1)    # [batch_size, n]
        fm_first_sparse = torch.sum(fm_first_sparse, 1, keepdim=True)         # [batch_size, 1]
        fm_first = fm_first_dense + fm_first_sparse                           # [batch_size ,1]

        '''FM二阶'''
        fm_second_sparse = torch.cat([self.sparse_second_emd[i](sparse_inputs[:, i])for i in range(sparse_inputs.shape[1])], -1)
        fm_second_sparse = fm_second_sparse.reshape(sparse_inputs.shape[0], sparse_inputs.shape[1], -1)        # [batch_size, n, emb_size]

        square_of_sum = torch.sum(fm_second_sparse, 1) ** 2
        sum_of_square = torch.sum(fm_second_sparse ** 2, 1)
        fm_second = square_of_sum - sum_of_square
        fm_second = 0.5 * torch.sum(fm_second, 1, keepdim=True)                # [batch_size, 1]

        '''DNN'''
        dnn_out = torch.flatten(fm_second_sparse, 1)                           # [batch_size, sparse_size]
        dnn_out = torch.cat([dense_inputs, dnn_out], 1)                        # [batch_size, dense_size + sparse_size]
        dnn_out = self.dnn(dnn_out)                                            # [batch_size, 1]

        output = fm_first + fm_second + dnn_out
        output = self.sigmoid(output)
        return output
