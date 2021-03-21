import gym
import time
import numpy as np


class QLearning(object):
    def __init__(self, n_states, n_actions, epsilon, gamma, lr, epsilon_decay, min_epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.qtable = self.build_q_table()

    def build_q_table(self):
        return np.zeros((self.n_states, self.n_actions))

    def choose_action(self, obs):
        if np.random.uniform() >= self.epsilon:
            Q_max = np.max(self.qtable[obs])
            action_list = np.where(self.qtable[obs] == Q_max)[0]
            action = np.random.choice(action_list)
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn_q_table(self, s, a, r, s_, done):
        q_predict = self.qtable[s][a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.qtable[s_])
        self.qtable[s][a] += self.lr * (q_target - q_predict)

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0', is_slippery=False)
    agent = QLearning(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        epsilon=0.95,
        gamma=0.9,
        lr=0.1,
        epsilon_decay=0.95,
        min_epsilon=0.01
    )

    render = False
    episodes = 500
# ===============================Train============================================
    for episode in range(episodes):

        total_reward, total_steps = 0, 0
        s = env.reset()
        if render:
            env.render()
            time.sleep(1)

        while True:
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            agent.learn_q_table(s, a, r, s_, done)
            s = s_
            total_reward += r
            total_steps += 1

            if render:
                env.render()
                time.sleep(1)

            if done:
                break

        if episode % 5 == 0:
            agent.update_epsilon()

        print('Episode {:03d} | Step:{:03d} | Reward:{:.03f}'.format(episode, total_steps, total_reward))
