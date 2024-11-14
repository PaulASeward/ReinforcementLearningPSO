import numpy as np


class DiscreteActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config
        self.action_methods = {
            0: self.do_nothing,
            1: self.increase_inertia,
            2: self.decrease_inertia,
            3: self.increase_social_factor,
            4: self.decrease_social_factor,
        }

        self.action_names = ['Do nothing',
                             'Increase inertia',
                             'Decrease inertia',
                             'Increase social factor',
                             'Decrease social factor', ]

    def __call__(self, action):
        if not isinstance(action, int):
            action = action.item()
        action_method = self.action_methods.get(action, lambda: None)
        action_method()

    def do_nothing(self):
        return

    def reset_all_particles_without_memory(self):
        self.swarm.reinitialize()

    def reset_all_particles_keep_global_best(self):
        old_gbest_pos = self.swarm.P[np.argmin(self.swarm.pbest_val)]
        old_gbest_val = np.min(self.swarm.pbest_val)

        self.swarm.reinitialize()

        # Keep Previous Solution before resetting.
        if old_gbest_val < self.swarm.gbest_val:
            self.swarm.gbest_pos = old_gbest_pos
            self.swarm.gbest_val = old_gbest_val

    def reset_all_particles_keep_personal_best(self):
        old_pbest_pos = self.swarm.P

        self.swarm.reinitialize()

        self.swarm.P = old_pbest_pos
        self.swarm.update_swarm_valuations_and_bests()

    def reset_slow_particles(self):
        avg_velocity = self._calculate_average_velocity()
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        replacement_positions = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF,
                                                  size=(self.swarm.swarm_size, self.swarm.dimension))
        replacement_velocities = np.full((self.swarm.swarm_size, self.swarm.dimension), 0)
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        self.swarm.X = np.where(slow_particles_reshaped, replacement_positions, self.swarm.X)
        self.swarm.V = np.where(slow_particles_reshaped, replacement_velocities, self.swarm.V)
        self.swarm.P = np.where(slow_particles_reshaped, self.swarm.X, self.swarm.P)

        self.swarm.update_swarm_valuations_and_bests()

    def increase_all_velocities(self):
        self.swarm.V = self.swarm.V * 1.10
        self.swarm.V = np.clip(self.swarm.V, -self.swarm.abs_max_velocity, self.swarm.abs_max_velocity)

    def decrease_all_velocities(self):
        self.swarm.V = self.swarm.V * 0.90

    def _calculate_average_velocity(self):
        self.swarm.update_velocity_maginitude()
        return np.mean(self.swarm.velocity_magnitudes)

    def increase_velocities_of_slow_velocities(self):
        avg_velocity = self._calculate_average_velocity()
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        faster_velocities = self.swarm.V * 1.10
        faster_velocities = np.clip(faster_velocities, -self.swarm.abs_max_velocity, self.swarm.abs_max_velocity)
        self.swarm.V = np.where(slow_particles_reshaped, faster_velocities, self.swarm.V)

    def decrease_velocities_of_fast_particles(self):
        avg_velocity = self._calculate_average_velocity()
        fast_particles = self.swarm.velocity_magnitudes > avg_velocity
        fast_particles_reshaped = fast_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        slower_velocities = self.swarm.V * 0.90
        self.swarm.V = np.where(fast_particles_reshaped, slower_velocities, self.swarm.V)

    def increase_max_velocity(self):
        self.swarm.abs_max_velocity *= 1.10
        self.swarm.abs_max_velocity = np.clip(self.swarm.abs_max_velocity, self.config.v_min, self.config.v_max)

    def decrease_max_velocity(self):
        self.swarm.abs_max_velocity *= 0.90
        self.swarm.abs_max_velocity = np.clip(self.swarm.abs_max_velocity, self.config.v_min, self.config.v_max)

    def increase_social_factor(self):
        self.swarm.c1 *= 1.10  # Social component
        self.swarm.c1 = np.clip(self.swarm.c1, self.config.c_min, self.config.c_max)

        self.swarm.c2 *= 0.90  # Cognitive component
        self.swarm.c2 = np.clip(self.swarm.c2, self.config.c_min, self.config.c_max)

    def increase_inertia(self):
        self.swarm.w *= 1.10
        self.swarm.w = np.clip(self.swarm.w, self.config.w_min, self.config.w_max)

    def decrease_inertia(self):
        self.swarm.w *= 0.90
        self.swarm.w = np.clip(self.swarm.w, self.config.w_min, self.config.w_max)

    def decrease_social_factor(self):
        self.swarm.c1 *= 0.90
        self.swarm.c1 = np.clip(self.swarm.c1, self.config.c_min, self.config.c_max)

        self.swarm.c2 *= 1.10
        self.swarm.c2 = np.clip(self.swarm.c2, self.config.c_min, self.config.c_max)

    # Threshold actions to promote exploration vs exploitation
    def decrease_gbest_replacement_threshold(self):
        self.swarm.gbest_replacement_threshold *= self.swarm.gbest_replacement_threshold_decay
        self.swarm.gbest_replacement_threshold = np.clip(self.swarm.gbest_replacement_threshold,
                                                         self.config.gbest_replacement_threshold_min,
                                                         self.config.gbest_replacement_threshold_max)

    def increase_gbest_replacement_threshold(self):
        self.swarm.gbest_replacement_threshold *= 1.10
        self.swarm.gbest_replacement_threshold = np.clip(self.swarm.gbest_replacement_threshold,
                                                         self.config.gbest_replacement_threshold_min,
                                                         self.config.gbest_replacement_threshold_max)

    def decrease_pbest_replacement_threshold(self):
        self.swarm.pbest_replacement_threshold *= self.swarm.pbest_replacement_threshold_decay
        self.swarm.pbest_replacement_threshold = np.clip(self.swarm.pbest_replacement_threshold,
                                                         self.config.pbest_replacement_threshold_min,
                                                         self.config.pbest_replacement_threshold_max)

    def increase_pbest_replacement_threshold(self):
        self.swarm.pbest_replacement_threshold *= 1.10
        self.swarm.pbest_replacement_threshold = np.clip(self.swarm.pbest_replacement_threshold,
                                                         self.config.pbest_replacement_threshold_min,
                                                         self.config.pbest_replacement_threshold_max)


class ContinuousActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config

        self.action_names = ['Inertia Param',
                             'Social Param',
                             'Cognitive Param']
        self.action_offset = [self.swarm.config.w, self.swarm.config.c1, self.swarm.config.c2]

    def __call__(self, action):
        """
        :param action: Tuple of 3 values representing the change in inertia, social, and cognitive parameters. Each value should be in the range [-1, 1]
        """
        action_with_offset = np.array(action) + self.action_offset
        self.swarm.w = action_with_offset[0]
        self.swarm.c1 = action_with_offset[1]
        self.swarm.c2 = action_with_offset[2]
