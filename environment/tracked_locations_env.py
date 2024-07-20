from environment.env import *
from environment.actions.actions import Actions
import csv

class TrackedLocationsPSOEnv(PSOEnv):
    def __init__(self, config):
        super().__init__(config=config)

        # Track Locations and Valuations
        self.meta_data_headers = ['Action', 'Action Name', 'Replacement Threshold', 'Global Best Gravity', 'Individual Best Gravity']
        self.track_locations = config.track_locations
        self._store_locations_and_valuations = False
        if self.track_locations:
            self.tracked_locations = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size, self._dim))
            self.tracked_velocities = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size, self._dim))
            self.tracked_best_locations = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size, self._dim))
            self.tracked_valuations = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size))
            self.meta_data = []
            self.env_swarm_locations_path = None
            self.env_swarm_velocities_path = None
            self.env_swarm_best_locations_path = None
            self.env_swarm_evaluations_path = None
            self.env_meta_data_path = None

        self._actions_count = 0
        self._episode_ended = False
        self._episode_actions = []
        self._episode_values = []
        self._best_fitness = None
        self.current_best_f = None

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

        # Save Locations and Valuations
        if self._store_locations_and_valuations:
            eps_tracked_locations, eps_tracked_velocities, eps_tracked_best_locations, eps_tracked_valuations = self.swarm.get_tracked_locations_and_valuations()
            self.tracked_locations[self._actions_count - 1] = eps_tracked_locations
            self.tracked_velocities[self._actions_count - 1] = eps_tracked_velocities
            self.tracked_best_locations[self._actions_count - 1] = eps_tracked_best_locations
            self.tracked_valuations[self._actions_count - 1] = eps_tracked_valuations
            self.meta_data.append([action_index, self.actions_descriptions[action_index], self.swarm.pbest_replacement_threshold, self.swarm.c1, self.swarm.c2])


        self.current_best_f = self.swarm.get_current_best_fitness()

        if self._best_fitness is None:
            reward = self._minimum - self.current_best_f
            self._best_fitness = self.current_best_f
        else:
            reward = max(self._best_fitness - self.current_best_f, 0)  # no penalty in reward
            # reward = self._minimum - self.current_best_f
            self._best_fitness = min(self._best_fitness, self.current_best_f)

        if self._episode_ended:
            # Save Locations and Valuations
            if self._store_locations_and_valuations:
                np.save(self.env_swarm_locations_path, self.tracked_locations)
                np.save(self.env_swarm_velocities_path, self.tracked_velocities)
                np.save(self.env_swarm_best_locations_path, self.tracked_best_locations)
                np.save(self.env_swarm_evaluations_path, self.tracked_valuations)
                with open(self.env_meta_data_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.meta_data_headers)
                    writer.writerows(self.meta_data)

                # Reset the arrays
                self.tracked_locations = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size, self._dim))
                self.tracked_velocities = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size, self._dim))
                self.tracked_best_locations = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size, self._dim))
                self.tracked_valuations = np.zeros((self._max_episodes, self._obs_per_episode, self._swarm_size))
                self.meta_data = []
                self._store_locations_and_valuations = False
                self.swarm.track_locations = False  # Turn off tracking

            return ts.termination(self._observation, reward)

        else:
            return ts.transition(self._observation, reward, discount=1.0)

    #   returns: TimeStep(step_type, reward, discount, observation)

    def store_locations_and_valuations(self, store: bool, env_swarm_locations_path=None, env_swarm_velocities_path=None, env_swarm_best_locations_path=None, env_swarm_evaluations_path=None, env_meta_data_path=None):
        """
        This setter-like method acts as a toggle with automatic save, turn off, and reset at the end of a terminating episode.
        """
        self._store_locations_and_valuations = store
        self.swarm.track_locations = store
        if store:  # New Directories are made for each new tracked episode
            self.env_swarm_locations_path = env_swarm_locations_path
            self.env_swarm_velocities_path = env_swarm_velocities_path
            self.env_swarm_best_locations_path = env_swarm_best_locations_path
            self.env_swarm_evaluations_path = env_swarm_evaluations_path
            self.env_meta_data_path = env_meta_data_path
