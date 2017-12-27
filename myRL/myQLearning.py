import collections
import numpy as np
import gym
import math


class QAgent(object):
    def __init__(self, action_space, eps, learning_rate, discount, max_step, discretize):
        self.action_space = action_space
        self.eps = eps
        self.discount = discount
        self.learning_rate = learning_rate
        self.max_step = max_step
        self.discretize = discretize
        self.q = collections.defaultdict(lambda: np.zeros(action_space.n))  # Q(x,a) = 0

    def act(self, state, eps=None):
        if eps is None or np.random.random() > eps:
            return np.argmax(self.q[state])
        else:
            return self.action_space.sample()  # 以eps概率随机选择action

    def learn(self, env):
        alpha = self.learning_rate
        gamma = self.discount

        state = self.discretize(env.reset())
        for t in range(self.max_step):
            action = self.act(state, self.eps)
            obs, reward, done, _ = env.step(action)
            next_state = self.discretize(obs)
            next_action = np.argmax(self.q[state])
            self.q[state][action] += alpha * (reward + gamma * self.q[next_state][next_action] - self.q[state][action])
            if not done:
                state = next_state
            else:
                # print("Episode finished after {} steps".format(t + 1))
                break


class CartPole(object):
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200

    def __init__(self, n_bin, min_value, max_value):
        self.n = len(n_bin)
        self.bin = [np.linspace(min_value[i], max_value[i], n_bin[i]) for i in range(self.n)]

    def discretize(self, obs):
        return tuple([int(np.digitize(obs[i], self.bin[i])) for i in range(self.n)])

    @staticmethod
    def learning_rate(t):
        return max(0.1, min(0.5, 1.0 - math.log10((t + 1) / 25)))

    def eps(self, t):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(- t / self.eps_decay)


def main():
    max_step = 20000
    gym.envs.register(
        id='CartPoleMyRL-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=max_step,
        reward_threshold=19500.0
    )
    env = gym.make('CartPoleMyRL-v0')
    cart_pole = CartPole(
        n_bin=(10, 10, 10, 10),
        min_value=(-2.0, -2.0, math.radians(-40), -2.0),
        max_value=(2.0, 2.0, math.radians(40), 2.0)
    )
    agent = QAgent(
        action_space=env.action_space,
        eps=0.1,
        learning_rate=0.01,
        discount=0.999,
        max_step=max_step,
        discretize=cart_pole.discretize
    )
    for t in range(5000):
        agent.eps = cart_pole.eps(t)
        # agent.learning_rate = cart_pole.learning_rate(t)
        agent.learn(env)

    print('Test start')
    out_dir = '/home/fuji/tmp/result/cartpole/cur'
    env = gym.wrappers.Monitor(env, directory=out_dir, force=True)
    for i in range(100):
        state = cart_pole.discretize(env.reset())
        for t in range(max_step):
            action = agent.act(state, 0)
            state, reward, done, _ = env.step(action)
            state = cart_pole.discretize(state)
            if done:
                print("Episode finished after {} steps".format(t + 1))
                break
    env.close()


if __name__ == '__main__':
    main()
