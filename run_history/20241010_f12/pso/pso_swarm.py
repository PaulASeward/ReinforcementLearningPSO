import numpy as np


class PSOSwarm:

    def __init__(self, objective_function, config):
        self.X = None  # Current Position of particles
        self.V = None  # Current Velocity of particles
        self.P = None  # Best Position of particles

        # PSO Parameters
        self.config = config
        self.w = config.w  # Inertia weight to prevent velocities becoming too large
        self.c1 = config.c1  # Social component Learning Factor
        self.c2 = config.c2  # Cognitive component Learning Factor
        self.function = objective_function
        self.dimension = config.dim
        self.swarm_size = config.swarm_size
        self.rangeF = config.rangeF

        # Threshold Params
        self.gbest_replacement_threshold, self.pbest_replacement_threshold = config.replacement_threshold, config.replacement_threshold
        self.gbest_replacement_threshold_decay, self.pbest_replacement_threshold_decay = config.replacement_threshold_decay, config.replacement_threshold_decay

        # Observation Parameters
        self.num_swarm_obs_intervals = config.num_swarm_obs_intervals
        self.swarm_obs_interval_length = config.swarm_obs_interval_length
        self.iterations = self.num_swarm_obs_intervals * self.swarm_obs_interval_length

        # Track Locations
        self.track_locations = config.track_locations
        if self.track_locations:
            self.tracked_locations = np.zeros((self.iterations, self.swarm_size, self.dimension))
            self.tracked_velocities = np.zeros((self.iterations, self.swarm_size, self.dimension))
            self.tracked_best_locations = np.zeros((self.iterations, self.swarm_size, self.dimension))
            self.tracked_valuations = np.zeros((self.iterations, self.swarm_size))
            self.track_locations = False

        # Set Constraints for clamping position and limiting velocity
        self.abs_max_velocity = self.rangeF
        self.abs_max_position = self.rangeF

        self.velocity_magnitudes = None
        self.relative_fitness = None
        self.average_batch_counts = None
        self.pbest_replacement_counts = None

        self.val = None
        self.gbest_val = None
        self.gbest_pos = None
        self.pbest_val = None

        # Initialize the swarm's positions velocities and best solutions
        self._initialize()

    def reinitialize(self):
        self._initialize()

    def _initialize(self):
        # Initialize 3 matrices for Current Position, Velocity, and Position of particles' best solution
        self.X = np.random.uniform(low=-1 * self.rangeF, high=self.rangeF, size=(self.swarm_size, self.dimension))
        self.V = np.full((self.swarm_size, self.dimension), 0)
        self.P = self.X

        # Reset the adjustable parameters to starting values
        self.w = self.config.w
        self.c1 = self.config.c1
        self.c2 = self.config.c2
        self.gbest_replacement_threshold, self.pbest_replacement_threshold = self.config.replacement_threshold, self.config.replacement_threshold
        self.abs_max_velocity = self.rangeF

        self.velocity_magnitudes = None
        self.relative_fitness = None
        self.average_batch_counts = None
        self.pbest_replacement_counts = None

        self.val = None
        self.gbest_val = None
        self.gbest_pos = None
        self.pbest_val = None

        # Static class variable to track pbest_val replacements
        self.pbest_replacement_counts = np.zeros(self.swarm_size)
        self.pbest_replacement_batchcounts = np.zeros((self.num_swarm_obs_intervals, self.swarm_size))
        self.average_batch_counts = np.zeros(self.swarm_size)

        # Record the initialized particle and global best solutions
        self.update_swarm_valuations_and_bests()

    def update_swarm_valuations_and_bests(self):
        self.val = self.eval(self.P)  # Vector of particle's current value of its position based on Function Eval
        self.pbest_val = self.val  # Vector of particle's best value of a visited position.
        self.gbest_pos = self.P[np.argmin(self.pbest_val)]  # Vector of globally best visited position.
        self.gbest_val = np.min(self.pbest_val)  # Single value of the global best position

        self.update_relative_fitness()
        self.update_velocity_maginitude()

    def update_relative_fitness(self):
        self.relative_fitness = (self.pbest_val - self.gbest_val) / np.abs(self.pbest_val)

    def update_velocity_maginitude(self):
        self.velocity_magnitudes = np.linalg.norm(self.V, axis=1)

    def update_batch_counts(self):
        # Calculate the average batch count for each particle
        self.average_batch_counts = np.mean(self.pbest_replacement_batchcounts, axis=0)

    def get_observation(self):
        self.update_relative_fitness()
        self.update_velocity_maginitude()
        self.update_batch_counts()

        return np.concatenate([self.velocity_magnitudes, self.relative_fitness, self.average_batch_counts], axis=0)

    def get_tracked_locations_and_valuations(self):
        return self.tracked_locations, self.tracked_velocities, self.tracked_best_locations, self.tracked_valuations

    def get_current_best_fitness(self):
        return self.gbest_val

    def eval(self, X):
        return self.function.Y_matrix(np.array(X).astype(float))

    def update_velocities(self, leader):
        # Generate social and cognitive components of velocity matrix update using np.random
        social = self.c1 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (leader - self.X)
        cognitive = self.c2 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (self.P - self.X)

        # Update new velocity with old velocity*inertia plus component matrices
        self.V = self.w * self.V + social + cognitive
        self.V = np.clip(self.V, -self.abs_max_velocity, self.abs_max_velocity)

    def update_position(self):
        # Clamp position inside boundary and reflect them in case they are out of the boundary based on:
        # S. Helwig, J. Branke, and S. Mostaghim, "Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization," IEEE TEC: 17(2), 2013, pp. 259-271

        self.X = self.X + self.V # Add velocity to current position

        # Identify positions outside the boundaries, and reflect the out-of-bounds positions
        out_of_bounds = np.logical_or(self.X < -self.abs_max_position, self.X > self.abs_max_position)
        self.X = np.where(out_of_bounds, np.sign(self.X) * 2 * self.abs_max_position - self.X, self.X)
        self.V = np.where(out_of_bounds, 0, self.V)  # Out-of-bounds velocities are set to 0

        # Use function evaluation for each particle (vector) in Swarm to provide value for each position in X.
        self.val = self.eval(self.X)

    def update_pbest_with_elitist_selection(self):
        improved_particles = self.val < self.pbest_val  # Update each Particle's best position for each particle index
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(improved_particles, self.val, self.pbest_val)

        self.pbest_replacement_counts += improved_particles  # Update pbest_val replacement counter

    def update_gbest_with_elitist_selection(self):
        self.gbest_pos = self.P[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)

    def optimize(self):
        for obs_interval_idx in range(self.num_swarm_obs_intervals):
            for i in range(self.swarm_obs_interval_length):
                self.update_velocities(self.gbest_pos)  # Input global leader particle position
                self.update_position()
                self.update_pbest_with_elitist_selection()
                # self.update_pbest_with_non_elitist_selection()
                self.update_gbest_with_elitist_selection()

                if self.track_locations:
                    self.tracked_locations[obs_interval_idx * self.swarm_obs_interval_length + i] = self.X
                    self.tracked_velocities[obs_interval_idx * self.swarm_obs_interval_length + i] = self.V
                    self.tracked_best_locations[obs_interval_idx * self.swarm_obs_interval_length + i] = self.P
                    self.tracked_valuations[obs_interval_idx * self.swarm_obs_interval_length + i] = self.val

            self.pbest_replacement_batchcounts[obs_interval_idx] = self.pbest_replacement_counts
            self.pbest_replacement_counts = np.zeros(self.swarm_size)

    def update_pbest_with_non_elitist_selection_random(self):
        pbest_change = (self.val - self.pbest_val) / np.abs(self.pbest_val)
        improved_particles = pbest_change < 0

        # Allow exploitative search, per standard for better solutions
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(improved_particles, self.val, self.pbest_val)
        self.pbest_replacement_counts += improved_particles

        # Promote Exploratory Search allowing for non-elitist selection
        non_elitist_improvements = np.logical_and(pbest_change >= 0, np.random.uniform(size=self.swarm_size) > self.pbest_replacement_threshold + pbest_change)
        self.P = np.where(non_elitist_improvements[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(non_elitist_improvements, self.val, self.pbest_val)
        self.pbest_replacement_counts += non_elitist_improvements

    def update_pbest_with_non_elitist_selection(self):
        improved_particles = self.pbest_replacement_threshold * self.val < self.pbest_val
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(improved_particles, self.val, self.pbest_val)

        self.pbest_replacement_counts += improved_particles  # Update pbest_val replacement counter






