from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import numpy as np
import Vectorized_PSOGlobalLocal as pso
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from typing import Any
from tf_agents.typing import types
import functions
import os


class PSOEnv(py_environment.PyEnvironment):
    def __init__(self, func_num, dimension, minimum):
        super().__init__()
        swarm_size = 50
        self._observ_size = swarm_size * 3  # [0-49]: Velocities, [50-99]: Relative Fitness, [100-149]: Average Replacement Rate
        self._dim = dimension
        self.func_num = func_num

        self._observation_interval = 1000
        self._replacement_ratio_interval = 100

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self._observ_size,), dtype=np.float64, name='observation')

        self._actions_count = 0
        self._episode_ended = False
        self._max_episodes = 10
        self._episode_actions = []
        csv_directory = "episode_actions/"
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)
        self.csv_filename = os.path.join(csv_directory, f"function{func_num}_.csv")

        self._max_evals = self._max_episodes * self._observation_interval
        self._best_fitness = None
        self._minimum = minimum

        obj_f = functions.CEC_functions(dimension)

        self.swarm = pso.PSOVectorSwarmGlobalLocal(obj_f, func_num, dimension, self._observation_interval, swarm_size, RangeF=100)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

            See `reset(self)` docstring for more details
        """
        self._actions_count = 0
        self._episode_ended = False
        self._episode_actions = []
        self._best_fitness = None

        # Restart the swarm with initializing criteria
        self.swarm.reinitialize()

        # Concatenate the three arrays into a single array
        self._observation = self.swarm.get_observation()

        # return ts.TimeStep(ts.StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._states)
        return ts.restart(self._observation)

    def _step(self, action):
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
          action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """

        if self._episode_ended:
            # Last action ended the episode, so we need to create a new episode:
            return self.reset()

        # Make sure episodes don't go on forever.
        self._actions_count += 1

        if self._actions_count == self._max_episodes:
            self._episode_ended = True

        # Implementation of the actions
        self._episode_actions.append(action)
        if action == 0:  # Do nothing special
            self.swarm.optimize()
            self._observation = self.swarm.get_observation()
            current_best_f = self.swarm.get_current_best_fitness()
        elif action == 1:  # Reset slower half
            self.swarm.reset_slow_particles()
            self.swarm.optimize()  # Reset all particles
            self._observation = self.swarm.get_observation()
            current_best_f = self.swarm.get_current_best_fitness()
        elif action == 2:  # Encourage social learning
            self.swarm.increase_social_factor()
            self.swarm.optimize()
            self._observation = self.swarm.get_observation()
            current_best_f = self.swarm.get_current_best_fitness()
        elif action == 3:  # Discourage social learning
            self.swarm.decrease_social_factor()
            self.swarm.optimize()
            self._observation = self.swarm.get_observation()
            current_best_f = self.swarm.get_current_best_fitness()
        elif action == 4:  # Reset all particles. Maybe keep global leader?
            self.swarm.reinitialize()
            self.swarm.optimize()
            self._observation = self.swarm.get_observation()
            current_best_f = self.swarm.get_current_best_fitness()

        if self._best_fitness is None:
            reward = self._minimum - current_best_f
            self._best_fitness = current_best_f
        else:
            reward = max(self._best_fitness - current_best_f, 0)  # no penalty in reward
            # reward = self._minimum - current_best_f
            self._best_fitness = min(self._best_fitness, current_best_f)

        if self._episode_ended:
            self.store_episode_actions_to_csv(self._episode_actions)
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward, discount=1.0)

    def store_episode_actions_to_csv(self, actionsRow):
        with open(self.csv_filename, mode='a', newline='') as csv_file:
            csv.writer(csv_file).writerow(actionsRow)

    # supposedly not needed
    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass
