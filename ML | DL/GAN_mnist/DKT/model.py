import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DKT, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        out, (hn, cn) = self.rnn(X)    # 如果h0和c0未给出，则默认为0
        out = self.fc(out)
        return self.sig(out)

