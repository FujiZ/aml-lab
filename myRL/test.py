import os

import torch
import numpy as np

import utils
import myDQN
import myImprovedDQN

# auto test for DQN & IDQN using current model

model_dir = 'model/'
result_dir = 'report/result/'


def test_dqn(helper, prefix, epoch):
    for i in range(1, 10):
        name = prefix + str(i)
        model_path = model_dir + name + '.pkl'
        if os.path.isfile(model_path):
            helper.agent.model.load_state_dict(torch.load(model_path))
            reward_list = []
            for t in range(epoch):
                reward_list.append(helper.play())
            with open(result_dir + prefix[:-1] + '.txt', 'a') as out:
                out.write(str(np.mean(reward_list)) + ', ' + str(np.std(reward_list)) + '\n')


def test_idqn(helper, prefix, epoch):
    for i in range(1, 10):
        name = prefix + str(i)
        model_path = model_dir + name + '.pkl'
        if os.path.isfile(model_path):
            helper.agent.eval_model.load_state_dict(torch.load(model_path))
            reward_list = []
            for t in range(epoch):
                reward_list.append(helper.play())
            with open(result_dir + prefix[:-1] + '.txt', 'a') as out:
                out.write(str(np.mean(reward_list)) + ', ' + str(np.std(reward_list)) + '\n')


def test_dqn_all():
    car = myDQN.MountainCar()
    bot = myDQN.Acrobot()
    test_dqn(car, 'car-dqn-', 100)
    test_dqn(bot, 'bot-dqn-', 100)


def test_idqn_all():
    car = myImprovedDQN.MountainCar()
    bot = myImprovedDQN.Acrobot()
    test_idqn(car, 'car-idqn-', 100)
    test_idqn(bot, 'bot-idqn-', 100)


if __name__ == '__main__':
    utils.register_env()
    for i in range(10):
        test_dqn_all()
        test_idqn_all()
