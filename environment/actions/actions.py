import numpy as np


class Actions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config

    def do_nothing(self):
        return

    def reset_all_particles(self):
        self.swarm.reinitialize()

    def reset_all_particles_keep_global_best(self):
        old_gbest_pos = self.swarm.P[np.argmin(self.swarm.pbest_val)]
        old_gbest_val = np.min(self.swarm.pbest_val)

        self.swarm.reinitialize()

        # Keep Previous Solution before resetting.
        if old_gbest_val < self.swarm.gbest_val:
            self.swarm.gbest_pos = old_gbest_pos
            self.swarm.gbest_val = old_gbest_val

    def reset_slow_particles(self):
        self.swarm.update_velocity_maginitude()
        avg_velocity = np.mean(self.swarm.velocity_magnitudes)
        slow_particles = self.swarm.velocity_magnitudes < avg_velocity
        replacement_positions = np.random.uniform(low=-1 * self.swarm.rangeF, high=self.swarm.rangeF, size=(self.swarm.swarm_size, self.swarm.dimension))
        replacement_velocities = np.full((self.swarm.swarm_size, self.swarm.dimension), 0)
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.swarm.X

        self.swarm.X = np.where(slow_particles_reshaped, replacement_positions, self.swarm.X)
        self.swarm.V = np.where(slow_particles_reshaped, replacement_velocities, self.swarm.V)
        self.swarm.P = self.swarm.X

        self.swarm.update_swarm_valuations_and_bests()

    def increase_social_factor(self):
        self.swarm.c1 *= 1.10  # Social component
        self.swarm.c1 = np.clip(self.swarm.c1, self.swarm.c_min, self.swarm.c_max)

        self.swarm.c2 *= 0.90  # Cognitive component
        self.swarm.c2 = np.clip(self.swarm.c2, self.swarm.c_min, self.swarm.c_max)

    def decrease_social_factor(self):
        self.swarm.c1 *= 0.90
        self.swarm.c1 = np.clip(self.swarm.c1, self.swarm.c_min, self.swarm.c_max)

        self.swarm.c2 *= 1.10
        self.swarm.c2 = np.clip(self.swarm.c2, self.swarm.c_min, self.swarm.c_max)

    # Threshold actions to promote exploration vs exploitation
    def decrease_gbest_replacement_threshold(self):
        self.swarm.gbest_replacement_threshold *= self.swarm.gbest_replacement_threshold_decay
        self.swarm.gbest_replacement_threshold = np.clip(self.swarm.gbest_replacement_threshold, self.swarm.gbest_replacement_threshold_min, self.swarm.gbest_replacement_threshold_max)

    def increase_gbest_replacement_threshold(self):
        self.swarm.gbest_replacement_threshold *= 1.10
        self.swarm.gbest_replacement_threshold = np.clip(self.swarm.gbest_replacement_threshold, self.swarm.gbest_replacement_threshold_min, self.swarm.gbest_replacement_threshold_max)

    def decrease_pbest_replacement_threshold(self):
        self.swarm.pbest_replacement_threshold *= self.swarm.pbest_replacement_threshold_decay
        self.swarm.pbest_replacement_threshold = np.clip(self.swarm.pbest_replacement_threshold, self.swarm.pbest_replacement_threshold_min, self.swarm.pbest_replacement_threshold_max)

    def increase_pbest_replacement_threshold(self):
        self.swarm.pbest_replacement_threshold *= 1.10
        self.swarm.pbest_replacement_threshold = np.clip(self.swarm.pbest_replacement_threshold, self.swarm.pbest_replacement_threshold_min, self.swarm.pbest_replacement_threshold_max)



