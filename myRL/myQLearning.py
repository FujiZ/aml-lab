import collections
import numpy as np
import gym
import math

import utils


class QAgent(object):
    def __init__(self, action_space, discretize,
                 discount=0.99, reward=lambda *arg: arg[1]):
        self.action_space = action_space
        self.discount = discount
        self.reward = reward
        self.discretize = discretize
        self.qtable = collections.defaultdict(lambda: np.zeros(action_space.n))  # Q(x,a) = 0

    def act(self, state, eps=None):
        if eps is None or np.random.random() > eps:
            return np.argmax(self.qtable[state])
        else:
            return self.action_space.sample()  # 以eps概率随机选择action

    def train(self, env, max_step, eps, learning_rate):
        gamma = self.discount

        state = self.discretize(env.reset())
        for t in range(max_step):
            action = self.act(state, eps)
            obs, reward, done, _ = env.step(action)
            reward = self.reward(obs, reward, done)
            next_state = self.discretize(obs)
            next_action = np.argmax(self.qtable[state])
            self.qtable[state][action] += learning_rate * (
                    reward + gamma * self.qtable[next_state][next_action] - self.qtable[state][action])
            if not done:
                state = next_state
            else:
                print("Train finished after {} steps".format(t + 1))
                break


class QHelper(object):
    def __init__(self, env, agent, max_step,
                 eps_start, eps_end, eps_decay,
                 lr_start, lr_end, lr_decay):
        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.lr_decay = lr_decay
        self.episodes_done = 0

    def train(self, episode):
        for t in range(episode):
            self.agent.train(self.env, self.max_step, self.eps(self.episodes_done),
                             self.learning_rate(self.episodes_done))
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

    def learning_rate(self, step):
        return self.lr_end + (self.lr_start - self.lr_end) * math.exp(-step / self.lr_decay)


class CartPole(QHelper):
    def __init__(self):
        env = gym.make('CartPoleMyRL-v0')
        super(CartPole, self).__init__(
            env=env,
            agent=QAgent(
                action_space=env.action_space,
                discretize=self.discretize,
                discount=0.99,
                reward=self.reward,
            ),
            max_step=20000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=200,
            lr_start=0.9,
            lr_end=0.0015,
            lr_decay=200,
        )

        bin_size = (2, 2, 7, 7)
        min_value = (-2.4, -1.1, math.radians(-41.8), -0.9)
        max_value = (2.4, 1.1, math.radians(41.8), 0.9)

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


class MountainCar(QHelper):
    def __init__(self):
        env = gym.make('MountainCarMyRL-v0')
        super(MountainCar, self).__init__(
            env=env,
            agent=QAgent(
                action_space=env.action_space,
                discretize=self.discretize,
                discount=0.99,
                reward=self.reward,
            ),
            max_step=2000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=200,
            lr_start=0.9,
            lr_end=0.0015,
            lr_decay=200,
        )

        bin_size = (20, 20)
        min_value = (-1.2, -0.07)
        max_value = (0.6, 0.07)

        self.n_state = env.observation_space.shape[0]
        self.bin = [np.linspace(min_value[i], max_value[i], bin_size[i]) for i in range(self.n_state)]

    def discretize(self, obs):
        return tuple([int(np.digitize(obs[i], self.bin[i])) for i in range(self.n_state)])

    @staticmethod
    def reward(obs, reward, done):
        if not done:
            return reward
        else:
            return 100


class Acrobot(QHelper):
    def __init__(self):
        env = gym.make('AcrobotMyRL-v0')
        super(Acrobot, self).__init__(
            env=env,
            agent=QAgent(
                action_space=env.action_space,
                discretize=self.discretize,
                discount=0.9,
                reward=self.reward,
            ),
            max_step=2000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=200,
            lr_start=0.9,
            lr_end=0.0015,
            lr_decay=200,
        )

        bin_size = (10, 10, 10, 10, 10, 10)
        min_value = env.observation_space.high
        max_value = env.observation_space.low

        self.n_state = env.observation_space.shape[0]
        self.bin = [np.linspace(min_value[i], max_value[i], bin_size[i]) for i in range(self.n_state)]

    def discretize(self, obs):
        return tuple([int(np.digitize(obs[i], self.bin[i])) for i in range(self.n_state)])

    @staticmethod
    def reward(obs, reward, done):
        if not done:
            return reward
        else:
            return 100


if __name__ == '__main__':
    utils.register_env()
    # pole = CartPole()
    # pole.train(1000)
    # car = MountainCar()
    # car.train(2000)
    acrobot = Acrobot()
    acrobot.train(1000)
