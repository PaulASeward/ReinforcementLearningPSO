import gymnasium as gym

gym.register(
    id="PsoGymEnv-v0",
    entry_point="environment.pso_gym_env:PsoGymEnv",
)