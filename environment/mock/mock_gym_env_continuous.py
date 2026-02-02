import gymnasium as gym
import numpy as np
from environment.actions.actions import Action


class MockContinuousPsoGymEnv(gym.Env):
    """Continuous environment."""

    # reward_range = (-float("inf"), float("inf"))

    def __init__(self, config, actions: Action):
        self._func_num = config.pso_config.func_num
        self._action_dimensions = config.env_config.action_dimensions
        self._minimum = config.pso_config.fDeltas[config.pso_config.func_num - 1]

        self._max_episodes = config.env_config.num_episodes
        self._num_swarm_obs_intervals = config.env_config.num_swarm_obs_intervals
        self._swarm_obs_interval_length = config.env_config.swarm_obs_interval_length
        self._obs_per_episode = config.env_config.obs_per_episode
        self._swarm_size = config.pso_config.swarm_size
        self._dim = config.pso_config.pso_dim

        self._observation_length = config.env_config.observation_length
        self.swarm = actions.swarm
        self.actions = actions
        self.action_space = actions.get_action_space()
        self.observation_space = actions.get_observation_space()

        self._actions_count = 0
        self._episode_ended = False
        self._episode_actions = []
        self._episode_values = []
        self._best_fitness = None
        self.current_best_f = None


    def _get_obs(self):
        # return self.swarm.get_observation()
        return np.random.rand(self._observation_length)

    def _get_reward(self):
        reward = np.random.rand() - 0.5  # Random reward between -0.5 and 0.5
        self._best_fitness = np.random.rand() * 100 + 400
        # self.current_best_f = self.swarm.get_current_best_fitness()
        #
        # if self._best_fitness is None:
        #     reward = self._minimum - self.current_best_f
        #     self._best_fitness = self.current_best_f
        # else:
        #     reward = max(self._best_fitness - self.current_best_f, 0)  # no penalty in reward
        #     # reward = self._minimum - self.current_best_f
        #     self._best_fitness = min(self._best_fitness, self.current_best_f)

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

        # # Implementation of the action
        # self.actions(action)
        # self.swarm.optimize()

        observation = self._get_obs()
        reward = self._get_reward()
        # truncated = False

        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, info

    def render(self, mode="human"):
        pass
