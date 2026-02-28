from typing import List, Callable, Dict
import numpy as np

from pso.pso_multiswarm import PSOSwarm


class Action:
    action_callback: Callable
    action_methods: Dict[int, Callable]
    action_names: List[str] = []
    lower_bound: np.ndarray = None
    upper_bound: np.ndarray = None
    practical_low_limit_action_space = []
    practical_high_limit_action_space = []

    def __call__(self, action, swarm: PSOSwarm):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError
