from typing import List, Callable, Optional
import gymnasium as gym

import numpy as np

from pso.pso_multiswarm import PSOMultiSwarm
from environment.actions.actions import Action
from pso.pso_swarm import PSOSwarm


class ContinuousMultiswarmActions(Action):
    def __init__(
            self,
            num_sub_swarms: int,
            action_callback: Callable,
            action_names: List[str],
            upper_bound: List[float],
            lower_bound: List[float],
            practical_action_high_limit: Optional[List[float]] = None,
            practical_action_low_limit: Optional[List[float]] = None,
    ):
        self.action_callback = action_callback
        self.num_sub_swarms = num_sub_swarms
        self.subswarm_action_dim = len(action_names)
        self.single_swarm_practical_action_high_limit = practical_action_high_limit
        self.single_swarm_practical_action_low_limit = practical_action_low_limit

        self.action_names = [
            f"SubSwarm {i + 1} {action_name}"
            for i in range(self.num_sub_swarms)
            for action_name in action_names
        ]

        if practical_action_high_limit is None:
            practical_action_high_limit = upper_bound
        if practical_action_low_limit is None:
            practical_action_low_limit = lower_bound

        self.practical_action_high_limit = [
            limit
            for _ in range(self.num_sub_swarms)
            for limit in practical_action_high_limit
        ]
        self.practical_action_low_limit = [
            limit
            for _ in range(self.num_sub_swarms)
            for limit in practical_action_low_limit
        ]

        self.lower_bound = np.array([
            lower_bound
            for _ in range(self.num_sub_swarms)
        ], dtype=np.float32).flatten()

        self.upper_bound = np.array([
            upper_bound
            for _ in range(self.num_sub_swarms)
        ], dtype=np.float32).flatten()

    def __call__(self, action, swarm: PSOMultiSwarm):
        # Ex.) Restructures the flattened action from size(config.num_sub_swarms * 3) to size (config.num_sub_swarms, 3)
        # reshaped_arr = action.reshape(3, 5)
        reformatted_action = np.array(action).reshape(self.num_sub_swarms, self.subswarm_action_dim)

        # Action should be a dedicated action for each subswarm
        for i, subswarm in enumerate(swarm.sub_swarms):
            self.action_callback(reformatted_action[i], subswarm, self.single_swarm_practical_action_high_limit, self.single_swarm_practical_action_low_limit)

    def get_action_space(self):
        return gym.spaces.Box(low=self.lower_bound, high=self.upper_bound,
                              shape=(len(self.action_names),), dtype=np.float32)


class ContinuousActions(Action):
    def __init__(
            self,
            action_callback: Callable,
            action_names: List[str],
            upper_bound: List[float],
            lower_bound: List[float],
            # We may want to have a practical limit for the action space that is different from the actual limit of the action space.
            # For example, the action space may allow for a wide range of values, but in practice, we may want to limit the actions
            # to a smaller range to ensure stable learning.
            practical_action_high_limit: Optional[List[float]] = None,
            practical_action_low_limit: Optional[List[float]] = None,
    ):
        self.action_names = action_names
        self.action_callback = action_callback

        if practical_action_high_limit is None:
            practical_action_high_limit = upper_bound
        if practical_action_low_limit is None:
            practical_action_low_limit = lower_bound

        self.practical_action_high_limit = practical_action_high_limit
        self.practical_action_low_limit = practical_action_low_limit

        self.lower_bound = np.array(lower_bound, dtype=np.float32)
        self.upper_bound = np.array(upper_bound, dtype=np.float32)

    def __call__(self, action, swarm: PSOSwarm):
        actions = np.array(action)
        self.action_callback(actions, swarm, self.practical_action_high_limit, self.practical_action_low_limit)

    def get_action_space(self):
        return gym.spaces.Box(low=self.lower_bound, high=self.upper_bound,
                              shape=(len(self.action_names),), dtype=np.float32)
