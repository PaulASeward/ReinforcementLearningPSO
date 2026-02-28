import gymnasium as gym
import numpy as np
from environment.actions.actions import Action
from environment.env_config import RLEnvConfig
from environment.reward_functions import reward_functions_mapping
from pso.pso_config import PSOConfig
from pso.pso_swarm import PSOSwarm


class PsoGymEnv(gym.Env):
    """ PSO Agent environment."""
    # reward_range = (-float("inf"), float("inf"))

    def __init__(
            self,
            pso_config: PSOConfig,
            env_config: RLEnvConfig,
            actions: Action,
            swarm: PSOSwarm
    ):
        self._action_dimensions = env_config.action_dimensions
        self._max_episodes = env_config.num_episodes
        self._append_last_action_to_observation = env_config.append_last_action_to_observation
        self._append_episode_completion_percentage_to_observation = env_config.append_episode_completion_percentage_to_observation

        self._use_discrete_env = env_config.use_discrete_env
        self._mock_data = pso_config.use_mock_data

        self.swarm = swarm
        self.actions = actions
        self.action_space = actions.get_action_space()

        self._observation_length = env_config.observation_length
        low_limits_obs_space = np.zeros(self._observation_length, dtype=np.float32)
        high_limits_obs_space = np.full(self._observation_length, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low_limits_obs_space, high=high_limits_obs_space, shape=(self._observation_length,), dtype=np.float32)

        self.last_action = 0 if self._use_discrete_env else np.zeros(self._action_dimensions, dtype=np.float32)
        self._actions_count = 0
        self._current_episode_percent = 0
        self._episode_ended = False
        self._episode_actions = []
        self._episode_values = []

        self.reward_function = reward_functions_mapping[env_config.reward_function](pso_config, env_config)

    def _get_obs(self):
        # return self.swarm.get_observation().astype(np.float32)
        observation = self.swarm.get_observation()

        if self._append_episode_completion_percentage_to_observation:
            observation = np.append(observation, self._current_episode_percent)

        # Round the last action in each dimension to the nearest integer and append
        # rounded_last_action = np.round(self.last_action)

        if self._append_last_action_to_observation:
            observation = np.append(observation, self.last_action)

        return observation.astype(np.float32)

    def _get_reward(self):
        current_best_f = self.swarm.get_current_best_fitness()
        reward = self.reward_function(current_best_f, self._actions_count)
        return np.float32(reward) if type(reward) is int else reward.astype(np.float32)

    def _get_done(self):
        return self._episode_ended

    def _get_info(self):
        info = self.swarm.get_swarm_observation()
        current_best_fitness = self.swarm.get_current_best_fitness()
        info["fitness_reward"] = self.reward_function.get_fitness_reward_for_plots(current_best_fitness)

        return info

    def reset(self, seed=None, return_info=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        current_best_f = self.swarm.get_current_best_fitness()
        self.reward_function.reset(current_best_f)

        self.last_action = np.zeros(self._action_dimensions, dtype=np.float32)
        self._actions_count = 0
        self._episode_ended = False
        self._current_episode_percent = 0

        # Restart the swarm with initializing criteria
        if not self._mock_data:
            self.swarm.reinitialize()

        observation = self._get_obs()
        info = self._get_info()
        self.reward_function._best_relative_fitness_for_plots = None  # Reset since get_info() uses it

        return observation, info

    def step(self, action):
        if self._episode_ended:
            print("Terminal episode reached unexpectedly. Please check the environment implementation")
            # Last action ended the episode, so we need to create a new episode:
            return self.reset()

        self._actions_count += 1
        self._current_episode_percent = self._actions_count / self._max_episodes
        if self._actions_count == self._max_episodes:
            self._episode_ended = True

        # Implementation of the action
        if not self._mock_data:
            self.actions(action, self.swarm)
            self.swarm.optimize()

            if self._use_discrete_env and not isinstance(action, int):
                action = action.item()

            self.last_action = action

        observation = self._get_obs()
        reward = self._get_reward()

        terminated = self._get_done()
        info = self._get_info()

        return observation, reward, terminated, info

    def render(self, mode="human"):
        pass
