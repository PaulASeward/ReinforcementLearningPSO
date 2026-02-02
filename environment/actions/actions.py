from typing import List

import numpy as np
import gymnasium as gym

from config import Config
from pso.pso_multiswarm import PSOSwarm


class Action:
    action_names: List[str] = []
    lower_bound: np.ndarray = None
    upper_bound: np.ndarray = None
    practical_low_limit_action_space = []
    practical_high_limit_action_space = []
    actual_low_limit_action_space = []
    actual_high_limit_action_space = []

    actions_descriptions: List[str] = []

    # Maybe? Either Box or Discrete Type
    action_space: gym.spaces.Discrete | gym.spaces.Box = None

    def __init__(self, swarm: PSOSwarm, config: Config):
        self.swarm = swarm
        self.config = config

    def __call__(self, action):
        raise NotImplementedError
