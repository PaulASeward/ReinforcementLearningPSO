from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pso.pso_swarm import PSOSwarm
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from typing import Any
from tf_agents.typing import types
import pso.cec_benchmark_functions as benchmark_functions


class PSOEnv(py_environment.PyEnvironment):
    def __init__(self, config):
        super().__init__()
        self._func_num = config.func_num
        self._num_actions = config.num_actions
        self.actions_descriptions = config.action_names[:self._num_actions]

        self._minimum = config.fDeltas[config.func_num - 1]

        self._max_episodes = config.num_episodes
        self._num_swarm_obs_intervals = config.num_swarm_obs_intervals
        self._swarm_obs_interval_length = config.swarm_obs_interval_length
        self._obs_per_episode = config.obs_per_episode
        self._swarm_size = config.swarm_size
        self._dim = config.dim

        self._observ_size = config.swarm_size * 3  # [0-49]: Velocities, [50-99]: Relative Fitness, [100-149]: Average Replacement Rate
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=config.num_actions-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self._observ_size,), dtype=np.float64, name='observation')

        self._actions_count = 0
        self._episode_ended = False
        self._episode_actions = []
        self._episode_values = []
        self._best_fitness = None
        self.current_best_f = None

        obj_f = benchmark_functions.CEC_functions(dim=config.dim, fun_num=config.func_num)

        self.swarm = PSOSwarm(objective_function=obj_f, config=config)

        self.action_methods = {
            0: lambda: None,
            1: self.swarm.decrease_pbest_replacement_threshold,  # Decrease Threshold for Replacement
            2: self.swarm.increase_pbest_replacement_threshold,  # Increase Threshold for Replacement
        }

        # self.action_methods = {
        #     0: lambda: None,  # Do nothing
        #     1: self.swarm.increase_social_factor,  # Encourage social learning
        #     2: self.swarm.decrease_social_factor,  # Discourage social learning
        #     3: self.swarm.reset_slow_particles,  # Reset slower half
        #     4: self.swarm.reset_all_particles_keep_global_best,  # Reset all particles. Keep global leader.
        # }

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

        self._actions_count += 1
        if self._actions_count == self._max_episodes:
            self._episode_ended = True

        # Implementation of the action
        action_index = action.item()
        action_method = self.action_methods.get(action_index, lambda: None)
        action_method()

        # Execute common operations after action
        self.swarm.optimize()
        self._observation = self.swarm.get_observation()


        self.current_best_f = self.swarm.get_current_best_fitness()

        if self._best_fitness is None:
            reward = self._minimum - self.current_best_f
            self._best_fitness = self.current_best_f
        else:
            reward = max(self._best_fitness - self.current_best_f, 0)  # no penalty in reward
            # reward = self._minimum - self.current_best_f
            self._best_fitness = min(self._best_fitness, self.current_best_f)

        if self._episode_ended:
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward, discount=1.0)

    #   returns: TimeStep(step_type, reward, discount, observation)

    # supposedly not needed
    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass
