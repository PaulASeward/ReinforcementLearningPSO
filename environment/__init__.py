import gym

gym.register(
    id="ContinuousPsoGymEnv-v0",
    entry_point="environment.gym_env_continuous:ContinuousPsoGymEnv",
    max_episode_steps=20,
)
