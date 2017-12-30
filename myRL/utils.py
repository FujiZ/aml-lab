import gym
import matplotlib.pyplot as plt
import numpy as np


def register_env():
    gym.envs.register(
        id='CartPoleMyRL-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=20000,
        reward_threshold=19995.0,
    )

    gym.envs.register(
        id='MountainCarMyRL-v0',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=2000,
        reward_threshold=-110.0,
    )

    gym.envs.register(
        id='AcrobotMyRL-v0',
        entry_point='gym.envs.classic_control:AcrobotEnv',
        max_episode_steps=2000,
    )


def plot(title, xlabel, ylabel, y):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(y)
    plt.show()


def plot_loss(title, in_file):
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Loss per Episode')
    plt.title(title)
    loss = np.load(in_file)
    plt.plot(range(1, len(loss) + 1), loss)
    plt.show()


def plot_reward(title, in_file):
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Reward per Episode')
    plt.title(title)
    reward = np.load(in_file)
    plt.plot(range(1, len(reward) + 1), reward)
    plt.show()
