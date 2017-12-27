import gym


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
