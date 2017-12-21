from collections import defaultdict
import numpy as np
import gym
from gym import wrappers


class QAgent(object):
    def __init__(self, action_space, eps, discount, learning_rate, epoch, discretizer):
        self.action_space = action_space
        self.eps = eps
        self.discount = discount
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.discretizer = discretizer
        self.q = defaultdict(lambda: np.zeros(action_space.n))  # init Q(x,a) to 0

    def act(self, obs, eps=None):
        # obs is discrete state
        if eps is None:
            eps = self.eps
        if np.random.random() > eps:
            return np.argmax(self.q[obs])  # 以1-eps概率按照policy选择
        else:
            return self.action_space.sample()  # 以eps概率随机选择action

    def learn(self, env):
        alpha = self.learning_rate
        gamma = self.discount

        obs = self.discretizer(env.reset())
        for t in range(1, self.epoch + 1):
            # action = self.act(obs, 1 / np.sqrt(t))
            action = self.act(obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = self.discretizer(next_obs)
            next_action = np.argmax(self.q[next_obs])
            if done:
                reward = -100
            self.q[obs][action] += alpha * (reward + gamma * self.q[next_obs][next_action] - self.q[obs][action])
            if not done:
                obs = next_obs
            else:
                break


class CartPole(object):
    def __init__(self, n_bin, min_value, max_value):
        self.n = len(n_bin)
        self.bin = [np.linspace(min_value[i], max_value[i], n_bin[i]) for i in range(self.n)]

    def discretize(self, obs):
        return tuple([int(np.digitize(obs[i], self.bin[i])) for i in range(self.n)])


def main():
    max_steps = 20000
    gym.envs.register(
        id='CartPoleMyRL-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=max_steps,
        reward_threshold=19500.0,
    )
    env = gym.make('CartPoleMyRL-v0')
    cart_pole = CartPole(
        n_bin=(8, 8, 8, 8),
        min_value=(-2.4, -2.0, -0.5, -3.0),
        max_value=(2.4, 2.0, 0.5, 3.0)
    )
    agent = QAgent(
        action_space=env.action_space,
        eps=0.5,
        discount=0.9,
        learning_rate=0.5,
        epoch=100,
        discretizer=cart_pole.discretize
    )
    for t in range(1, 1001):
        # agent.learning_rate = 1 / np.sqrt(t+1)
        agent.eps = 1 / np.sqrt(t)
        agent.learn(env)

    out_dir = '/home/fuji/tmp/cartpole'
    env = wrappers.Monitor(env, directory=out_dir, force=True)
    obs = cart_pole.discretize(env.reset())
    for t in range(max_steps):
        action = agent.act(obs, 0)
        obs, reward, done, _ = env.step(action)
        obs = cart_pole.discretize(obs)
        if done:
            print("Episode finished after {} steps".format(t + 1))
            break


if __name__ == '__main__':
    main()
