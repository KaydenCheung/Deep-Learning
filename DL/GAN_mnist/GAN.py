import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_dim = 100
batch_size = 128
num_epoch = 10


# =================================================生成器================================================================
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.br1(self.fc1(x))
        x = self.br2(self.fc2(x))
        x = x.reshape(-1, 128, 7, 7)
        x = self.conv1(x)
        output = self.conv2(x)
        return output


# =================================================判别器================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.pl1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.pl2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pl1(x)
        x = self.conv2(x)
        x = self.pl2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output

# ==================================================训练================================================================
def training(x):
    '''判别器'''
    real_x = x.to(device)
    real_output = D(real_x)
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))

    fake_x = G(torch.randn([batch_size, input_dim]).to(device)).detach()
    fake_output = D(fake_x)
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))

    loss_D = real_loss + fake_loss

    optim_D.zero_grad()
    loss_D.backward()
    optim_D.step()

    '''生成器'''
    fake_x = G(torch.randn([batch_size, input_dim]).to(device))
    fake_output = D(fake_x)
    loss_G = loss_func(fake_output, torch.ones_like(fake_output).to(device))

    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    return loss_D, loss_G


if __name__ == '__main__':
    train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    G = Generator(input_dim).to(device)
    D = Discriminator().to(device)
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002)
    loss_func = nn.BCELoss()

    for epoch in range(num_epoch):
        total_loss_D, total_loss_G = 0, 0
        for i, (x, _) in enumerate(train_loader):
            loss_D, loss_G = training(x)

            total_loss_D += loss_D
            total_loss_G += loss_G

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print('Epoch {:02d} | Step {:04d} / {} | Loss_D {:.4f} | Loss_G {:.4f}'.format(epoch, i + 1, len(train_loader), total_loss_D / (i + 1), total_loss_G / (i + 1)))

        x = torch.randn(64, input_dim).to(device)
        img = G(x)
        save_image(img, './data/results/' + '%d_epoch.png' % epoch)
