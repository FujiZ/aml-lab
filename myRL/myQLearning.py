import collections
import numpy as np
import gym
import math

import utils


class QAgent(object):
    def __init__(self, action_space, discretize,
                 discount=0.99, learning_rate=0.001,
                 reward=lambda *arg: arg[1]):
        self.action_space = action_space
        self.discount = discount
        self.learning_rate = learning_rate
        self.reward = reward
        self.discretize = discretize
        self.qtable = collections.defaultdict(lambda: np.zeros(action_space.n))  # Q(x,a) = 0

    def act(self, state, eps=None):
        if eps is None or np.random.random() > eps:
            return np.argmax(self.qtable[state])
        else:
            return self.action_space.sample()  # 以eps概率随机选择action

    def train(self, env, max_step, eps):
        alpha = self.learning_rate
        gamma = self.discount

        state = self.discretize(env.reset())
        for t in range(max_step):
            action = self.act(state, eps)
            obs, reward, done, _ = env.step(action)
            reward = self.reward(obs, reward, done)
            next_state = self.discretize(obs)
            next_action = np.argmax(self.qtable[state])
            self.qtable[state][action] += alpha * (
                    reward + gamma * self.qtable[next_state][next_action] - self.qtable[state][action])
            if not done:
                state = next_state
            else:
                print("Train finished after {} steps".format(t + 1))
                break


class QHelper(object):
    def __init__(self, env, agent, max_step,
                 eps_start, eps_end, eps_decay):
        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.episodes_done = 0

    def train(self, episode):
        for t in range(episode):
            self.agent.train(self.env, self.max_step, self.eps(self.episodes_done))
            self.episodes_done += 1

    def play(self):
        state = self.agent.discretize(self.env.reset())
        for t in range(self.max_step):
            action = self.agent.act(state)
            state, reward, done, _ = self.env.step(action)
            if not done:
                state = self.agent.discretize(state)
            else:
                print("Play finished after {} steps".format(t + 1))
                break

    def eps(self, step):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-step / self.eps_decay)


class CartPole(QHelper):
    def __init__(self):
        env = gym.make('CartPoleMyRL-v0')
        super(CartPole, self).__init__(
            env=env,
            agent=QAgent(
                action_space=env.action_space,
                discretize=self.discretize,
                discount=0.999,
                learning_rate=0.001,
                reward=self.reward,
            ),
            max_step=20000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=100,
        )

        bin_size = (4, 4, 20, 8)
        min_value = (-2.4, -0.5, math.radians(-41.8), math.radians(-50))
        max_value = (2.4, 0.5, math.radians(41.8), math.radians(50))

        self.n_state = env.observation_space.shape[0]
        self.bin = [np.linspace(min_value[i], max_value[i], bin_size[i]) for i in range(self.n_state)]

    def discretize(self, obs):
        return tuple([int(np.digitize(obs[i], self.bin[i])) for i in range(self.n_state)])

    @staticmethod
    def reward(obs, reward, done):
        if not done:
            return reward
        else:
            return -100


if __name__ == '__main__':
    utils.register_env()
    pole = CartPole()
    pole.train(1000)
