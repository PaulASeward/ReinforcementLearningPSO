from typing import List, Callable
import gymnasium as gym

import numpy as np

from pso.pso_multiswarm import PSOSubSwarm, PSOMultiSwarm
from environment.actions.actions import Action
from pso.pso_swarm import PSOSwarm


class ContinuousMultiswarmActions(Action):
    def __init__(
            self,
            swarm: PSOMultiSwarm,
            action_callback: Callable,
            action_names: List[str],
            practical_action_high_limit: List[float],
            practical_action_low_limit: List[float],
            actual_action_high_limit: List[float],
            actual_action_low_limit: List[float],
    ):
        super(Action, self).__init__(swarm)
        self.num_sub_swarms = len(swarm.sub_swarms)
        self.subswarm_action_dim = len(action_names)

        self.subswarm_actions = [
            ContinuousActions(sub_swarm, action_callback, action_names, practical_action_high_limit,
                              practical_action_low_limit, actual_action_high_limit, actual_action_low_limit) for
            sub_swarm in swarm.sub_swarms]

        self.action_names = [
            f"SubSwarm {i + 1} {action_name}"
            for i, subswarm in enumerate(self.subswarm_actions)
            for action_name in subswarm.action_names
        ]

        self.practical_action_high_limit = [
            limit
            for subswarm in self.subswarm_actions
            for limit in subswarm.practical_action_high_limit
        ]
        self.practical_action_low_limit = [
            limit
            for subswarm in self.subswarm_actions
            for limit in subswarm.practical_action_low_limit
        ]

        self.set_limits()

    def __call__(self, action):
        # Restructure the flattened action from size(config.num_sub_swarms * 3) to size (config.num_sub_swarms, 3)
        # reshaped_arr = action.reshape(3, 5)
        reformatted_action = np.array(action).reshape(self.num_sub_swarms, self.subswarm_action_dim)

        # Action should be a dedicated action for each subswarm
        for i, subswarm_action in enumerate(self.subswarm_actions):
            subswarm_action(reformatted_action[i])

    def set_limits(self):
        self.lower_bound = np.array([
            subswarm.actual_low_limit_action_space
            for subswarm in self.subswarm_actions
        ], dtype=np.float32).flatten()

        self.upper_bound = np.array([
            subswarm.actual_high_limit_action_space
            for subswarm in self.subswarm_actions
        ], dtype=np.float32).flatten()


    def get_action_space(self):
        return gym.spaces.Box(low=self.lower_bound, high=self.upper_bound,
                              shape=(len(self.action_names),), dtype=np.float32)

    def get_observation_space(self):
        low_limits_obs_space = np.zeros(self.observation_length,
                                        dtype=np.float32)  # 150-dimensional array with all elements set to 0
        high_limits_obs_space = np.full(self.observation_length, np.inf, dtype=np.float32)

        return gym.spaces.Box(low=low_limits_obs_space, high=high_limits_obs_space,
                              shape=(self.observation_length,), dtype=np.float32)


class ContinuousActions(Action):
    def __init__(
            self,
            swarm: PSOSwarm,
            action_callback: Callable,
            action_names: List[str],
            practical_action_high_limit: List[float],
            practical_action_low_limit: List[float],
            actual_action_high_limit: List[float],
            actual_action_low_limit: List[float]
    ):
        super(Action, self).__init__(swarm)
        self.action_names = action_names
        self.action_callback = action_callback
        self.practical_action_high_limit = practical_action_high_limit
        self.practical_action_low_limit = practical_action_low_limit
        self.actual_action_high_limit = actual_action_high_limit
        self.actual_action_low_limit = actual_action_low_limit

        self.set_limits()

    def __call__(self, action):
        actions = np.array(action)
        self.action_callback(actions, self.practical_action_high_limit, self.practical_action_low_limit)

    def set_limits(self):
        self.lower_bound = np.array(self.actual_low_limit_action_space, dtype=np.float32)
        self.upper_bound = np.array(self.actual_high_limit_action_space, dtype=np.float32)

    def reset_all_particles_keep_global_best(self):
        old_gbest_pos = self.swarm.P[np.argmin(self.swarm.P_vals)]
        old_gbest_val = np.min(self.swarm.P_vals)

        self.swarm.reinitialize()
        if type(self.swarm) == PSOSubSwarm:
            self.swarm.share_information_with_global_swarm = True

        # Keep Previous Solution before resetting.
        if old_gbest_val < self.swarm.gbest_val:
            self.swarm.gbest_pos = old_gbest_pos
            self.swarm.gbest_val = old_gbest_val
