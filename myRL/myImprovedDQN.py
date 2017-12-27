import gym
import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Variable

import myDQN

TARGET_REPLACE_C = 10


class ImprovedDQNAgent(object):
    def __init__(self, action_space, observation_space,
                 memory_size, batch_size, hidden_dim,
                 eps=0.1, discount=0.99, learning_rate=0.001,
                 reward=lambda *arg: arg[1]):
        self.action_space = action_space
        self.memory = myDQN.ReplayMemory(memory_size)
        self.batch_size = batch_size
        # eval_model is Q; target_model is Q^
        self.eval_model = myDQN.DQN(observation_space.shape[0], action_space.n, hidden_dim)
        self.target_model = myDQN.DQN(observation_space.shape[0], action_space.n, hidden_dim)
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.learns_done = 0
        self.eps = eps
        self.discount = discount
        self.reward = reward
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def act(self, x, eps=None):
        if eps is None:
            eps = self.eps
        if random.random() > eps:
            self.eval_model.train(mode=False)
            actions_value = self.eval_model(Variable(x.unsqueeze(0), volatile=True)).data.numpy()
            return np.argmax(actions_value)
        else:
            return self.action_space.sample()

    def train(self, env, max_step):
        state = torch.FloatTensor(env.reset())
        for t in range(max_step):
            action = self.act(state, self.eps)
            next_state, reward, done, _ = env.step(action)
            reward = self.reward(next_state, reward, done)
            if not done:
                next_state = torch.FloatTensor(next_state)
            else:
                # an empty tensor indicating a terminal state
                next_state = torch.FloatTensor()
            self.memory.push(state, action, reward, next_state)
            self.learn()
            if not done:
                state = next_state
            else:
                print("Learn finished after {} steps".format(t + 1))
                break

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
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
        next_state_values[torch.from_numpy(non_final_mask.astype(np.uint8))] = \
            self.target_model(non_final_next_state_batch).data.max(1)[0]

        # y_j
        expected_state_action_values = reward_batch + (self.discount * next_state_values.unsqueeze(1))
        expected_state_action_values.volatile = False

        loss = self.loss_func(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learns_done += 1
        if self.learns_done % TARGET_REPLACE_C == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())


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
                discount=0.999,
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


if __name__ == '__main__':
    cart_pole_max_step = 20000
    gym.envs.register(
        id='CartPoleMyRL-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=cart_pole_max_step,
        reward_threshold=19995.0,
    )
    mountain_car_max_step = 2000
    gym.envs.register(
        id='MountainCarMyRL-v0',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=mountain_car_max_step,
        reward_threshold=-110.0,
    )
    acrobot_max_step = 2000
    gym.envs.register(
        id='AcrobotMyRL-v0',
        entry_point='gym.envs.classic_control:AcrobotEnv',
        max_episode_steps=acrobot_max_step,
    )
    cart_pole = CartPole()
    cart_pole.train(100)
    # mountain_car = MountainCar()
    # mountain_car.train(10)
    # acrobot = Acrobot()
    # acrobot.train(10)
    # for i in range(100):
    #  cart_pole.test()
