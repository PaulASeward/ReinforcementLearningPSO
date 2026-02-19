from typing import List, Callable

import numpy as np
import gymnasium as gym

from pso.pso_multiswarm import PSOSwarm


class Action:
    action_callback: Callable
    action_names: List[str] = []
    observation_length: int = None
    lower_bound: np.ndarray = None
    upper_bound: np.ndarray = None
    practical_low_limit_action_space = []
    practical_high_limit_action_space = []
    actual_low_limit_action_space = []
    actual_high_limit_action_space = []

    actions_descriptions: List[str] = []

    # Maybe? Either Box or Discrete Type
    action_space: gym.spaces.Discrete | gym.spaces.Box = None

    def __init__(self, swarm: PSOSwarm):
        self.swarm = swarm
        self.observation_length = swarm.get_observation().shape[0]
        x=1

    def __call__(self, action):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError
