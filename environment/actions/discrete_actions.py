from typing import List, Dict, Callable

import numpy as np
from pso.pso_multiswarm import PSOMultiSwarm
import gymnasium as gym
from environment.actions.actions import Action


class DiscreteMultiswarmActions(Action):
    def __init__(self, swarm: PSOMultiSwarm, action_names: List[str], action_methods: Dict[int, Callable]):
        super(Action, self).__init__(swarm)

        self.subswarm_actions = [DiscreteActions(sub_swarm, action_names, action_methods) for sub_swarm in swarm.sub_swarms]
        self.action_names = ['Do Nothing'] + [
            f"SubSwarm {i + 1} {action_name}"
            for i, subswarm in enumerate(self.subswarm_actions)
            for action_name in subswarm.action_names
        ]

    def __call__(self, action):
        # Do nothing
        if action == 0:
            return
        action -= 1

        # Dividend operation to get the action for the respective subswarm
        subswarm_index = action // len(self.subswarm_actions[0].action_names)
        action_index = action % len(self.subswarm_actions[0].action_names)

        # Call the respective subswarm action
        self.subswarm_actions[subswarm_index](action_index)


class DiscreteActions(Action):
    def __init__(self, swarm, action_names: List[str], action_methods: Dict[int, Callable]):
        super(Action, self).__init__(swarm)
        self.action_names = action_names
        self.action_methods = action_methods

    def __call__(self, action):
        if not isinstance(action, int):
            action = action.item()
        action_method = self.action_methods.get(action, lambda: None)
        action_method()

    def get_action_space(self):
        return gym.spaces.Discrete(len(self.action_names))

    def get_observation_space(self):
        low_limits_obs_space = np.zeros(self.observation_length)  # 150-dimensional array with all elements set to 0
        high_limits_obs_space = np.full(self.observation_length, np.inf)

        return gym.spaces.Box(low=low_limits_obs_space, high=high_limits_obs_space,
                                                shape=(self.observation_length,), dtype=np.float32)
