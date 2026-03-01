from typing import List, Dict, Callable

from pso.pso_multiswarm import PSOMultiSwarm
import gymnasium as gym
from environment.actions.actions import Action
from pso.pso_swarm import PSOSwarm


class DiscreteMultiswarmActions(Action):
    def __init__(self, num_sub_swarms: int, action_names: List[str], action_methods: Dict[int, Callable]):
        self.num_sub_swarms = num_sub_swarms
        self.subswarm_num_actions = len(action_names)
        self.action_methods = action_methods

        self.action_names = ['Do Nothing'] + [
            f"SubSwarm {i + 1} {action_name}"
            for i in range(self.num_sub_swarms)
            for action_name in action_names
        ]

    def __call__(self, action, swarm: PSOMultiSwarm):
        # Do nothing
        if action == 0:
            return
        action -= 1

        # Dividend operation to get the action for the respective subswarm
        subswarm_index = action // self.subswarm_num_actions
        action_index = action % self.subswarm_num_actions

        subswarm = swarm.sub_swarms[subswarm_index]
        action_method = self.action_methods.get(action_index, lambda: None)
        action_method(subswarm)

    def get_action_space(self):
        return gym.spaces.Discrete(len(self.action_names))


class DiscreteActions(Action):
    def __init__(self, action_names: List[str], action_methods: Dict[int, Callable]):
        self.action_names = action_names
        self.action_methods = action_methods

    def __call__(self, action, swarm: PSOSwarm):
        if not isinstance(action, int):
            action = action.item()
        action_method = self.action_methods.get(action, lambda: None)
        action_method(swarm)

    def get_action_space(self):
        return gym.spaces.Discrete(len(self.action_names))
