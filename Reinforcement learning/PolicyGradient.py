import gym
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_states, n_actions, emb_dim):
        super(Net, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.emb_dim = emb_dim
        self.fc = nn.Linear(self.n_states, self.emb_dim)
        self.out = nn.Linear(self.emb_dim, self.n_actions)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class PG(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.lr = args.lr

        self.log_a = []
        self.ep_r = []

        self.model = Net(self.n_states, self.n_actions, args.emb_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)           # [1, n_states]
        logits = self.model(s)                                 # [1, n_actions]
        probs = F.softmax(logits, 1)
        action = torch.multinomial(probs, 1)                   # 根据概率采样
        self.log_a.append(torch.log(probs[0][action].squeeze(0)))    # 保存公式中的log值
        return action.item()

    def store(self, r):
        self.ep_r.append(r)

    def learn(self):
        processed_ep_r = np.zeros_like(self.ep_r)
        sum = 0
        for i in reversed(range(0, len(self.ep_r))):        # 回溯
           sum = sum * self.gamma + self.ep_r[i]
           processed_ep_r[i] = sum

        eps = np.finfo(np.float32).eps.item()
        processed_ep_r = (processed_ep_r - np.mean(processed_ep_r)) / (np.std(processed_ep_r) + eps)    # 归一化
        processed_ep_r = torch.FloatTensor(processed_ep_r)

        loss = -torch.sum(torch.cat(self.log_a) * processed_ep_r)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_a = []      # 数据使用完后即丢弃
        self.ep_r = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    env = gym.make('CartPole-v1')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    agent = PG(n_states, n_actions, args)

    x, y = [], []

    for episode in range(args.n_episodes):

        ep_reward, s = 0, env.reset()

        while True:
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            agent.store(r)

            ep_reward += r
            s = s_
            if done:
                break

        agent.learn()
        print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_reward))

        x.append(episode)
        y.append(ep_reward)

    plt.plot(x, y)
    plt.show()


