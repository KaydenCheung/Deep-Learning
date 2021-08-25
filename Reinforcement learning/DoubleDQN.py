import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
from collections import deque
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_states, n_actions, emb_dim):
        super(Net, self).__init__()
        self.fc = nn.Linear(n_states, emb_dim)
        self.out = nn.Linear(emb_dim, n_actions)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):
    def __init__(self, n_states, n_actions, args):
        self.n_states = n_states
        self.n_actions = n_actions
        self.emb_dim = args.emb_dim
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay = args.epsilon_decay
        self.gamma = args.gamma
        self.memory = deque(maxlen=args.memory_size)
        self.memory_count = 0

        self.eval_net = Net(self.n_states, self.n_actions, self.emb_dim)
        self.target_net = Net(self.n_states, self.n_actions, self.emb_dim)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)           # 这里是单个s，因为要进入神经网络计算，所以加一层维度
        if np.random.uniform() >= self.epsilon:
            actions_value = self.eval_net(s)
            action = np.argmax(actions_value.squeeze().detach()).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))
        self.memory_count += 1

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def update_q_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        batch_data = random.sample(self.memory, self.batch_size)
        s = torch.FloatTensor([d[0] for d in batch_data])          # [batch_size, 4]
        a = torch.LongTensor([d[1] for d in batch_data])           # [batch_size]
        r = [d[2] for d in batch_data]
        s_ = torch.FloatTensor([d[3] for d in batch_data])         # [batch_size, 4]
        done = [d[4] for d in batch_data]

        q_eval = self.eval_net(s).gather(1, a.reshape(-1, 1))      # [batch_size, 1]
        q_next = np.argmax(self.eval_net(s_).detach(), 1)          # [batch_size]
        q_target = self.target_net(s_)

        y = torch.zeros((self.batch_size, 1))
        for i in range(len(done)):
            target = r[i]
            if not done[i]:
                target += self.gamma * q_target[i][q_next[i]].item()
            y[i][0] = target

        loss = self.loss_func(q_eval.reshape(-1), y.reshape(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--memory_size', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--min_epsilon', type=float, default=0.02)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    agent = DQN(n_states, n_actions, args)
    is_render = False

    reward_count = 0
    update_count = 0
    x, y = [], []

    for episode in range(args.episodes):
        total_reward, loss = 0, 0
        s = env.reset()
        while True:
            if is_render:
                env.render()
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)

            total_reward += r

            agent.store(s, a, r, s_, done)

            if agent.memory_count > args.batch_size:
                agent.learn()

                update_count += 1
                agent.update_epsilon()
                if update_count % 20 == 0:
                    agent.update_q_target()

            s = s_
            if done:
                break

        x.append(episode)
        y.append(total_reward)

        # if total_reward >= 200:       # 当200分以上出现10次以上时，显示图像
        #     reward_count += 1
        #     if reward_count >= 10:
        #         is_render = True

        print('Episode {:03d} | Reward:{:.03f}'.format(episode, total_reward))

    plt.plot(x, y)
    plt.show()
