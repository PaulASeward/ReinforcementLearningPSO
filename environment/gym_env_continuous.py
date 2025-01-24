import gymnasium as gym
import numpy as np
from pso.cec_benchmark_functions import CEC_functions
from environment.actions.continuous_actions import ContinuousActions
from pso.pso_swarm import PSOSwarm


class ContinuousPsoGymEnv(gym.Env):
    """Continuous environment."""

    # reward_range = (-float("inf"), float("inf"))

    def __init__(self, config):
        self._func_num = config.func_num
        self._action_dimensions = config.action_dimensions
        self._minimum = config.fDeltas[config.func_num - 1]

        self._max_episodes = config.num_episodes
        self._num_swarm_obs_intervals = config.num_swarm_obs_intervals
        self._swarm_obs_interval_length = config.swarm_obs_interval_length
        self._obs_per_episode = config.obs_per_episode
        self._swarm_size = config.swarm_size
        self._dim = config.dim

        self._observation_length = config.observation_length
        low_limits_obs_space = np.zeros(self._observation_length, dtype=np.float32)  # 150-dimensional array with all elements set to 0
        high_limits_obs_space = np.full(self._observation_length, np.inf, dtype=np.float32)

        self.action_space = gym.spaces.Box(low=config.lower_bound, high=config.upper_bound, shape=(self._action_dimensions,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low_limits_obs_space, high=high_limits_obs_space, shape=(self._observation_length,), dtype=np.float32)

        self.swarm = PSOSwarm(objective_function=CEC_functions(dim=config.dim, fun_num=config.func_num), config=config)
        self.actions = ContinuousActions(swarm=self.swarm, config=config)
        config.actions_descriptions = self.actions.action_names[:self._action_dimensions]
        config.continuous_action_offset = self.actions.action_offset

        self._actions_count = 0
        self._episode_ended = False
        self._episode_actions = []
        self._episode_values = []
        self._best_fitness = None
        self.current_best_f = None


    def _get_obs(self):
        return self.swarm.get_observation().astype(np.float32)

    def _get_reward(self):
        self.current_best_f = self.swarm.get_current_best_fitness()

        if self._best_fitness is None:
            reward = self._minimum - self.current_best_f
            self._best_fitness = self.current_best_f
        else:
            reward = max(self._best_fitness - self.current_best_f, 0)  # no penalty in reward
            # reward = self._minimum - self.current_best_f
            self._best_fitness = min(self._best_fitness, self.current_best_f)

        return reward.astype(np.float32)

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
        self.swarm.reinitialize()

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

        # Implementation of the action
        self.actions(action)
        self.swarm.optimize()

        observation = self._get_obs()
        reward = self._get_reward()
        # truncated = False

        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, info

    def render(self, mode="human"):
        pass
