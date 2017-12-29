import gym
import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Variable

import myDQN
import utils


class ImprovedDQNAgent(object):
    def __init__(self, action_space, observation_space,
                 memory_size, batch_size, hidden_dim, target_c,
                 discount=0.99, learning_rate=0.001,
                 reward=lambda *arg: arg[1]):
        self.action_space = action_space
        self.memory = myDQN.ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.target_c = target_c
        # eval_model is Q; target_model is Q^
        self.eval_model = myDQN.DQN(observation_space.shape[0], action_space.n, hidden_dim)
        self.target_model = myDQN.DQN(observation_space.shape[0], action_space.n, hidden_dim)
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.learns_done = 0
        self.discount = discount
        self.reward = reward
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        # plot related args
        self.loss_list = []
        self.reward_list = []

    def act(self, x, eps=None):
        if eps is None or random.random() > eps:
            self.eval_model.train(mode=False)
            actions_value = self.eval_model(Variable(x.unsqueeze(0), volatile=True)).data.numpy()
            return np.argmax(actions_value)
        else:
            return self.action_space.sample()

    def train(self, env, max_step, eps):
        loss = 0.0
        total_reward = 0.0
        learn_count = 0
        state = torch.FloatTensor(env.reset())
        for t in range(max_step):
            action = self.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            reward = self.reward(next_state, reward, done)
            if not done:
                next_state = torch.FloatTensor(next_state)
            else:
                # an empty tensor indicating a terminal state
                next_state = torch.FloatTensor()
            self.memory.push(state, action, reward, next_state)
            if len(self.memory) >= self.batch_size:
                loss += self.learn()
                learn_count += 1
            if not done:
                state = next_state
            else:
                print("Train finished after {} steps".format(t + 1))
                if learn_count > 0:
                    self.loss_list.append(loss / learn_count)
                self.reward_list.append(total_reward)
                break

    def learn(self):
        self.eval_model.train(mode=True)

        batch = np.array(self.memory.sample(self.batch_size), dtype=object)

        state_batch = Variable(torch.stack(batch[:, 0]))
        action_batch = Variable(torch.from_numpy(batch[:, 1].astype(dtype=np.int64)).unsqueeze(1))
        reward_batch = Variable(torch.from_numpy(batch[:, 2].astype(dtype=np.float32))).unsqueeze(1)
        # Q(s_j, a_j)
        state_action_values = self.eval_model(state_batch).gather(1, action_batch)

        # select non final samples
        non_final_mask = np.vectorize(lambda x: x.dim() != 0)(batch[:, 3])
        non_final_next_state_batch = Variable(torch.stack(batch[non_final_mask, 3]))

        # let non_final_next_state_batch to go through model
        # max_a'Q^(s_{t+1},a')
        next_state_values = Variable(torch.zeros(self.batch_size).type(torch.FloatTensor))
        next_state_values.volatile = True
        next_state_values[torch.from_numpy(non_final_mask.astype(np.uint8))] = \
            self.target_model(non_final_next_state_batch).max(1)[0]

        # y_j
        expected_state_action_values = reward_batch + (self.discount * next_state_values.unsqueeze(1))

        loss = self.loss_func(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learns_done += 1
        if self.learns_done % self.target_c == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        return float(loss)


class CartPole(myDQN.DQNHelper):
    def __init__(self):
        env = gym.make('CartPoleMyRL-v0')
        super(CartPole, self).__init__(
            env=env,
            agent=ImprovedDQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                memory_size=10000,
                batch_size=128,
                hidden_dim=50,
                target_c=10,
                discount=0.99,
                learning_rate=0.001,
                reward=self.reward,
            ),
            max_step=20000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=200,
        )

    @staticmethod
    def reward(obs, reward, done):
        if not done:
            return reward
        else:
            return -100


class MountainCar(myDQN.DQNHelper):
    def __init__(self):
        env = gym.make('MountainCarMyRL-v0')
        super(MountainCar, self).__init__(
            env=env,
            agent=ImprovedDQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                memory_size=10000,
                batch_size=128,
                hidden_dim=50,
                target_c=10,
                discount=0.99,
                learning_rate=0.001,
                reward=self.reward,
            ),
            max_step=2000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=50,
        )

    @staticmethod
    def reward(obs, reward, done):
        if not done:
            return reward
        else:
            return 100


class Acrobot(myDQN.DQNHelper):
    def __init__(self):
        env = gym.make('AcrobotMyRL-v0')
        super(Acrobot, self).__init__(
            env=env,
            agent=ImprovedDQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                memory_size=10000,
                batch_size=128,
                hidden_dim=50,
                target_c=5,
                discount=0.99,
                learning_rate=0.001,
                reward=self.reward,
            ),
            max_step=2000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=200,
        )

    @staticmethod
    def reward(obs, reward, done):
        if not done:
            return reward
        else:
            return 100


if __name__ == '__main__':
    utils.register_env()
    # pole = CartPole()
    # pole.train(200)
    # car = MountainCar()
    # car.train(10)
    bot = Acrobot()
    bot.train(10)
