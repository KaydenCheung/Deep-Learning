import gym
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.action_layer = nn.Sequential(
            nn.Linear(self.n_states, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_actions)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(self.n_states, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        action = F.softmax(self.action_layer(x))
        value = self.value_layer(x)
        return action, value


class AC(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(AC, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = args.gamma
        self.lr = args.lr

        self.model = Net(self.n_states, self.n_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        probs, _ = self.model(s)
        action = torch.multinomial(probs, 1)        # 根据概率采样
        return action.item(), probs.squeeze(0)

    def critic_learn(self, s, s_, r, done):
        # if done:
        #     r -= 20
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        s_ = torch.unsqueeze(torch.FloatTensor(s_), 0)
        r = torch.FloatTensor([r])

        _, v = self.model(s)                # [1, 1]
        _, v_ = self.model(s_)              # [1, 1]
        v_ = v_.detach()
        v, v_ = v.squeeze(0), v_.squeeze(0)

        target = r
        if not done:
            target += self.gamma * v_

        loss_func = nn.MSELoss()
        loss = loss_func(v, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        advantage = (target - v).detach()
        return advantage

    def actor_learn(self, advantage, s, a):
        _, probs = self.choose_action(s)
        log_prob = probs.log()[a]

        loss = -advantage * log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    agent = AC(n_states, n_actions, args)

    x, y = [], []
    for episode in range(args.n_episodes):
        ep_reward, s= 0, env.reset()
        while True:
            a, _ = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            ep_reward += r

            advantage = agent.critic_learn(s, s_, r, done)
            agent.actor_learn(advantage, s, a)

            s = s_
            if done:
                break

        print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_reward))

        x.append(episode)
        y.append(ep_reward)

    plt.plot(x, y)
    plt.show()
