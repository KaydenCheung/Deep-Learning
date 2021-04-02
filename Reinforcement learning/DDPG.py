import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
from collections import deque
import matplotlib.pyplot as plt


# ==================================================Net==================================================================

class ActorNet(nn.Module):
    def __init__(self, n_states, n_actions, bound, emb_dim):
        super(ActorNet, self).__init__()
        self.bound = bound
        self.fc = nn.Linear(n_states, emb_dim)
        self.out = nn.Linear(emb_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = torch.tanh(self.out(x))    # tanh将值映射至[-1,1]
        action = x * self.bound
        return action


class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions, emb_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(n_states, emb_dim)
        self.fca = nn.Linear(n_actions, emb_dim)
        self.fcx = nn.Linear(emb_dim * 2, emb_dim)
        self.out = nn.Linear(emb_dim, 1)

    def forward(self, x, y):               # 输入状态和动作
        s = self.fcs(x)
        a = self.fca(y)
        x = torch.cat([s, a], 1)           # 拼接状态和动作
        x = F.relu(self.fcx(x))
        action_value = self.out(x)         # 计算Q(s,a)
        return action_value

# ==================================================DDPG=================================================================


class DDPG(object):
    def __init__(self, n_states, n_actions, bound, args):
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.memory = deque(maxlen=args.memory_size)

        self.actor_eval = ActorNet(n_states, n_actions, bound, self.emb_dim)
        self.actor_target = ActorNet(n_states, n_actions, bound, self.emb_dim)
        self.critic_eval = CriticNet(n_states, n_actions, self.emb_dim)
        self.critic_target = CriticNet(n_states, n_actions, self.emb_dim)

        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=0.001)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=0.002)

        self.loss_func = nn.MSELoss()

        self.init_para()

    def init_para(self):
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        action = self.actor_eval(s)
        return action.detach()

    def store(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def update_target_model(self):
        tmp_dict = {}
        for name, param in self.actor_target.named_parameters():
            tmp_dict[name] = (1-0.99) * self.actor_target.state_dict()[name] + 0.99 * self.actor_eval.state_dict()[name]
        self.actor_target.load_state_dict(tmp_dict)

        tmp_dict = {}
        for name, param in self.critic_target.named_parameters():
            tmp_dict[name] = (1 - 0.99) * self.critic_target.state_dict()[name] + 0.99 * self.critic_eval.state_dict()[name]
        self.critic_target.load_state_dict(tmp_dict)

    def learn(self):
        batch_data = random.sample(self.memory, self.batch_size)
        bs = torch.FloatTensor([d[0] for d in batch_data])            # torch.size([batch_size, n_states])
        ba = torch.FloatTensor([d[1] for d in batch_data])            # torch.size([batch_size, n_actions])
        br = torch.FloatTensor([d[2] for d in batch_data])            # [batch_size]
        bs_ = torch.FloatTensor([d[3] for d in batch_data])           # torch.size([batch_size, n_states])

        a = self.actor_eval(bs)                                       # torch.size([batch,n_actions])
        q = self.critic_eval(bs, a)                                   # torch.size([batch_size, 1)]
        loss = -torch.mean(q)                                         # 因为actor输出的a是使Q值最大的a，所有q值应该尽可能大，由于reward是负值，所有这里loss添加负号

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        a_ = self.actor_target(bs_)                                    # torch.size([batch,n_actions])
        q_ = self.critic_target(bs_, a_)                               # torch.size([batch_size, 1)]
        q_target = br.reshape(-1, 1) + self.gamma * q_                 # torch.size([batch_size, 1)]
        q_v = self.critic_eval(bs, ba)                                 # torch.size([batch_size, 1)]
        td_error = self.loss_func(q_target, q_v)

        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()

# =================================================Main=================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--len_episode', type=int, default=200)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--var', type=float, default=3.0)
    parser.add_argument('--var_decay', type=float, default=0.9995)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    env = gym.make('Pendulum-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    bound = env.action_space.high[0]
    agent = DDPG(n_states, n_actions, bound, args)

    var = args.var
    all_ep_r = []

    for episode in range(args.episodes):
        ep_r = 0
        s = env.reset()
        for t in range(args.len_episode):
            a = agent.choose_action(s)
            a = np.clip(np.random.normal(a, var), -bound, bound)    # 正态分布抽样
            s_, r, done, _ = env.step(a)
            agent.store(s, a, r, s_)

            if len(agent.memory) >= args.memory_size:
                var *= args.var_decay
                agent.learn()
                agent.update_target_model()

            ep_r += r
            s = s_

        print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_r))

        if episode == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)          # 平滑

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.show()




