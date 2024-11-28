import gym

gym.register(
    id="MockContinuousPsoGymEnv-v0",
    entry_point="environment.mock.mock_gym_env_continuous:MockContinuousPsoGymEnv",
)

gym.register(
    id="MockContinuousPmsoGymEnv-v0",
    entry_point="environment.mock.mock_gym_multiswarm_env_continuous:MockContinuousPmsoGymEnv",
)

gym.register(
    id="MockDiscretePsoGymEnv-v0",
    entry_point="environment.mock.mock_gym_env_discrete:MockDiscretePsoGymEnv",
)
