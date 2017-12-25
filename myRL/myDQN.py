import math
import numpy as np
import random
import collections
import gym

import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DQN(nn.Module):
    def __init__(self, n_action, n_observation):
        super(DQN, self).__init__()
        self.fc = nn.Linear(n_observation, 50)
        self.fc.weight.data.normal_(0, 0.1)
        self.head = nn.Linear(50, n_action)
        self.head.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc(x)
        x = functional.relu(x)
        return self.head(x)


class DQNAgent(object):
    def __init__(self, action_space, observation_space, eps,
                 learning_rate, discount, batch_size, max_step):
        self.action_space = action_space
        self.eps = eps
        self.discount = discount
        self.batch_size = batch_size
        self.max_step = max_step
        self.model = DQN(action_space.n, observation_space.shape[0])
        self.memory = ReplayMemory(2000)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def act(self, x, eps=None):
        if eps is None:
            eps = self.eps
        if random.random() > eps:
            actions_value = self.model(Variable(x.unsqueeze(0)))
            return np.argmax(actions_value.data.numpy())
        else:
            return self.action_space.sample()

    def learn(self, env):
        state = torch.FloatTensor(env.reset())
        for t in range(self.max_step):
            action = self.act(state, self.eps)
            next_state, reward, done, _ = env.step(action)
            if not done:
                next_state = torch.FloatTensor(next_state)
            else:
                # an empty tensor indicating a terminal state
                next_state = torch.FloatTensor()
                reward = -100
            self.memory.push(state, action, reward, next_state)
            self.optimize_model()
            if not done:
                state = next_state
            else:
                print("Episode finished after {} steps".format(t + 1))
                break

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.array(self.memory.sample(self.batch_size), dtype=object)

        state_batch = Variable(torch.stack(batch[:, 0]))
        action_batch = Variable(torch.from_numpy(batch[:, 1].astype(dtype=np.int64)).unsqueeze(1))
        reward_batch = Variable(torch.from_numpy(batch[:, 2].astype(dtype=np.float32))).unsqueeze(1)
        # Q(s_j, a_j)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # select non final entries
        non_final_mask = np.vectorize(lambda x: x.dim() != 0)(batch[:, 3])
        non_final_next_state_batch = Variable(torch.stack(batch[non_final_mask, 3]))

        # let non_final_next_state_batch to go through model
        # max_a'Q(s_{t+1},a')
        next_state_values = Variable(torch.zeros(self.batch_size).type(torch.FloatTensor))
        next_state_values[torch.from_numpy(non_final_mask.astype(np.uint8))] =\
            self.model(non_final_next_state_batch).data.max(1)[0]

        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        # next_state_values.volatile = False

        # y_j
        expected_state_action_values = reward_batch + (self.discount * next_state_values.unsqueeze(1))

        loss = self.loss_func(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    max_step = 20000
    gym.envs.register(
        id='CartPoleMyRL-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=max_step,
        reward_threshold=19995.0
    )
    env = gym.make('CartPoleMyRL-v0')
    agent = DQNAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        eps=0.9,
        learning_rate=0.01,
        discount=0.999,
        max_step=max_step,
        batch_size=128
    )
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    for t in range(200):
        agent.eps = eps_end + (eps_start - eps_end) * \
                        math.exp(-1. * t / eps_decay)
        agent.learn(env)

    out_dir = '/home/fuji/tmp/result/cartpole/cur'
    env = gym.wrappers.Monitor(env, directory=out_dir, force=True)
    state = torch.FloatTensor(env.reset())
    for t in range(max_step):
        action = agent.act(state, 0)
        state, reward, done, _ = env.step(action)
        if not done:
            state = torch.FloatTensor(state)
        else:
            print("Episode finished after {} steps".format(t + 1))
            break
    env.close()


if __name__ == '__main__':
    main()
