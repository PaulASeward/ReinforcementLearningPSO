import gymnasium as gym
import numpy as np

from environment.env_config import RLEnvConfig
from pso.pso_config import PSOConfig
from pso.pso_swarm import PSOSwarm
from environment.actions.actions import Action


class MockDiscretePsoGymEnv(gym.Env):
    """Continuous environment."""

    # reward_range = (-float("inf"), float("inf"))

    def __init__(
            self,
            pso_config: PSOConfig,
            env_config: RLEnvConfig,
            actions: Action,
            swarm: PSOSwarm
    ):
        self._func_num = pso_config.func_num
        self._num_actions = env_config.num_actions
        self._minimum = pso_config.fDeltas[pso_config.func_num - 1]

        self._max_episodes = env_config.num_episodes
        self._swarm_size = pso_config.swarm_size
        self._dim = pso_config.pso_dim

        self.swarm = swarm
        self.actions = actions
        self.action_space = actions.get_action_space()
        self._observation_length = env_config.observation_length

        # Does the dtype need to be specified here like continuous env?
        low_limits_obs_space = np.zeros(self._observation_length)
        high_limits_obs_space = np.full(self._observation_length, np.inf)
        self.observation_space = gym.spaces.Box(low=low_limits_obs_space, high=high_limits_obs_space,
                                                shape=(self._observation_length,), dtype=np.float32)

        self._actions_count = 0
        self._episode_ended = False
        self._episode_actions = []
        self._episode_values = []
        self._best_fitness = None
        self.current_best_f = None


    def _get_obs(self):
        return np.random.rand(self._observation_length)

    def _get_reward(self):
        reward = np.random.rand() - 0.5  # Random reward between -0.5 and 0.5
        self._best_fitness = np.random.rand() * 100 + 400
        return reward

    def _get_done(self):
        return self._episode_ended

    def _get_info(self):
        return self.swarm.get_swarm_observation()

    def reset(self, seed=None, return_info=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._actions_count = 0
        self._episode_ended = False
        self._best_fitness = None

        # Restart the swarm with initializing criteria
        # self.swarm.reinitialize()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if self._episode_ended:
            # Last action ended the episode, so we need to create a new episode:
            return self.reset()

        self._actions_count += 1
        if self._actions_count == self._max_episodes:
            self._episode_ended = True

        observation = self._get_obs()
        reward = self._get_reward()
        # truncated = False

        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, info

    def render(self, mode="human"):
        pass
