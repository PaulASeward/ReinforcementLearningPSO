import gym

gym.register(
    id="ContinuousPsoGymEnv-v0",
    entry_point="environment.gym_env_continuous:ContinuousPsoGymEnv",
    max_episode_steps=20,
)

gym.register(
    id="DiscretePsoGymEnv-v0",
    entry_point="environment.gym_env_discrete:DiscretePsoGymEnv",
    max_episode_steps=20,
)
