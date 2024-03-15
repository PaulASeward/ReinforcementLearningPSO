from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import numpy as np
from environment.pso_swarm import PSOSwarm
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from typing import Any
from tf_agents.typing import types
import environment.functions as functions
import os


class MockPSOEnv(py_environment.PyEnvironment):
    def __init__(self, config):
        super().__init__()
        self._num_actions = config.num_actions
        self._observ_size = config.swarm_size * 3  # Adjusted to match your mock data
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=config.num_actions - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._observ_size,), dtype=np.float64, name='observation')

        self._max_episodes = config.num_episodes
        self._episode_ended = False
        self._actions_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._actions_count = 0
        # Generate a random initial observation
        initial_observation = np.random.rand(self._observ_size)
        return ts.restart(initial_observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._actions_count += 1
        if self._actions_count >= self._max_episodes:
            self._episode_ended = True

        # Generate a random observation
        observation = np.random.rand(self._observ_size)
        # Generate a mock reward
        reward = np.random.rand() - 0.5  # Random reward between -0.5 and 0.5

        if self._episode_ended:
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward, discount=1.0)

    # The following methods might be required by the framework even if not used
    def get_info(self) -> Any:
        return None

    def get_state(self) -> Any:
        return None

    def set_state(self, state: Any) -> None:
        pass