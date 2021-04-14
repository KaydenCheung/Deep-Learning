import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, N, K):
        super(FM, self).__init__()
        self.w = nn.Linear(N, 1, bias=True)
        self.v = nn.Parameter(torch.rand(K, N))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fm_first = self.w(x).squeeze()    # [batch_size]
        fm_second = 0.5 * torch.sum(torch.pow(torch.matmul(x, self.v.t()), 2) - torch.matmul(torch.pow(x, 2), torch.pow(self.v.t(), 2)), 1)     # [batch_size]
        output = self.sigmoid(fm_first + fm_second)
        return output