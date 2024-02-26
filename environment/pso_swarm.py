import numpy as np
from sklearn.cluster import KMeans

class PSOSwarm:

    def __init__(self, objective_function, num_swarm_obs_intervals, swarm_obs_interval_length, dimension=30, swarm_size=50, RangeF=100):
        self.X = None  # Current Position of particles
        self.V = None  # Current Velocity of particles
        self.P = None  # Best Position of particles

        # Store Function Evaluator and current evaluation column vector for each particle's best.
        self.w = 0.729844  # Inertia weight to prevent velocities becoming too large
        self.c1 = 2.05 * self.w  # Social component Learning Factor
        self.c2 = 2.05 * self.w  # Cognitive component Learning Factor
        self.c_min = 0.88  # Min of 5 actions of decreasing 10%
        self.c_max = 2.41  # Max of 5 actions of increasing 10%

        self.gbest_replacement_threshold_min = 0.5
        self.gbest_replacement_threshold_max = 1.0
        self.gbest_replacement_threshold = 1.0  # Threshold to replace pbest with gbest
        self.gbest_replacement_threshold_decay = 0.95

        self.function = objective_function
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.rangeF = RangeF
        self.num_swarm_obs_intervals = num_swarm_obs_intervals
        self.swarm_obs_interval_length = swarm_obs_interval_length
        self.iterations = num_swarm_obs_intervals * swarm_obs_interval_length

        # Set Constraints for clamping position and limiting velocity
        self.Vmin, self.Vmax = -1 * RangeF, RangeF
        self.Xmin, self.Xmax = -1 * RangeF, RangeF

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

    def get_current_best_fitness(self):
        return self.gbest_val

    def reset_all_particles(self):
        self._initialize()

    def reset_all_particles_keep_global_best(self):
        old_gbest_pos = self.P[np.argmin(self.pbest_val)]
        old_gbest_val = np.min(self.pbest_val)

        self._initialize()

        # Keep Previous Solution before resetting.
        if old_gbest_val < self.gbest_val:
            self.gbest_pos = old_gbest_pos
            self.gbest_val = old_gbest_val

    def reset_slow_particles(self):
        self.update_velocity_maginitude()
        avg_velocity = np.mean(self.velocity_magnitudes)
        slow_particles = self.velocity_magnitudes < avg_velocity
        replacement_positions = np.random.uniform(low=-1 * self.rangeF, high=self.rangeF, size=(self.swarm_size, self.dimension))
        replacement_velocities = np.full((self.swarm_size, self.dimension), 0)
        slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match self.X

        self.X = np.where(slow_particles_reshaped, replacement_positions, self.X)
        self.V = np.where(slow_particles_reshaped, replacement_velocities, self.V)
        self.P = self.X

        self.update_swarm_valuations_and_bests()

    def increase_social_factor(self):
        self.c1 *= 1.10  # Social component
        self.c1 = np.clip(self.c1, self.c_min, self.c_max)

        self.c2 *= 0.90  # Cognitive component
        self.c2 = np.clip(self.c2, self.c_min, self.c_max)

    def decrease_social_factor(self):
        self.c1 *= 0.90
        self.c1 = np.clip(self.c1, self.c_min, self.c_max)

        self.c2 *= 1.10
        self.c2 = np.clip(self.c2, self.c_min, self.c_max)

    def decrease_gbest_replacement_threshold(self):
        self.gbest_replacement_threshold *= self.gbest_replacement_threshold_decay
        self.gbest_replacement_threshold = np.clip(self.gbest_replacement_threshold, self.gbest_replacement_threshold_min, self.gbest_replacement_threshold_max)

    def increase_gbest_replacement_threshold(self):
        self.gbest_replacement_threshold *= 1.10
        self.gbest_replacement_threshold = np.clip(self.gbest_replacement_threshold, self.gbest_replacement_threshold_min, self.gbest_replacement_threshold_max)

    def eval(self, X):
        return self.function.Y_matrix(np.array(X).astype(float))

    def update_velocities(self, leader):
        # Generate social and cognitive components of velocity matrix update using np.random
        social = self.c1 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (leader - self.X)
        cognitive = self.c2 * np.random.uniform(size=(self.swarm_size, self.dimension)) * (self.P - self.X)

        # Update new velocity with old velocity*inertia plus component matrices
        self.V = self.w * self.V + social + cognitive

        # Maintain velocity constraints
        self.V = np.clip(self.V, self.Vmin, self.Vmax)
        return

    def update_position(self):
        self.X = self.X + self.V # Add velocity to current position

        # Clamp position inside boundary and:
        # Reflect them in case they are out of the boundary based on: S. Helwig, J. Branke, and
        # S. Mostaghim, "Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization,"
        # IEEE TEC: 17(2), 2013, pp. 259-271

        # Identify positions outside the boundaries
        out_of_bounds = np.logical_or(self.X < self.Xmin, self.X > self.Xmax)

        # Reflect the out-of-bounds positions and set their velocities to zero
        self.X = np.where(out_of_bounds, np.sign(self.X) * 2 * self.Xmax - self.X, self.X)

        # Update the positions and evaluate the new positions
        self.V = np.where(out_of_bounds, 0, self.V)

        # Use function evaluation for each particle (vector) in Swarm to provide value for each position in X.
        self.val = self.eval(self.X)

    def update_pbest(self):
        improved_particles = self.val < self.pbest_val # Update each Particle's best position for each particle dimension
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(improved_particles, self.val, self.pbest_val)

        self.pbest_replacement_counts += improved_particles  # Update pbest_val replacement counter

    def select_diverse_global_best(self):
        # Number of clusters is arbitrarily set; adjust based on problem dimensionality or swarm size
        num_clusters = min(self.swarm_size // 5, 10)

        # Perform clustering based on particle positions
        if self.swarm_size > num_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(self.P)
            labels = kmeans.labels_
        else:
            labels = np.zeros(self.swarm_size)  # Fallback to no clustering if swarm is too small

        # Calculate cluster fitness: average or best fitness in each cluster
        cluster_fitness = np.array([self.pbest_val[labels == i].min() for i in range(num_clusters)])

        # Select the cluster with the highest potential for exploration
        # Here, potential is inversely related to the fitness; adjust the criterion as needed
        selected_cluster = np.argmax(cluster_fitness)

        # Update the global best with a particle from the selected cluster
        candidates_indices = np.where(labels == selected_cluster)[0]
        best_candidate_index = candidates_indices[np.argmin(self.pbest_val[candidates_indices])]
        self.gbest_pos = self.P[best_candidate_index]
        self.gbest_val = self.pbest_val[best_candidate_index]

    def update_gbest(self):
        self.gbest_pos = self.P[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)

    def optimize(self):
        for obs_interval_idx in range(self.num_swarm_obs_intervals):
            for _ in range(self.swarm_obs_interval_length):
                self.update_velocities(self.gbest_pos)  # Input global leader particle position
                self.update_position()
                self.update_pbest()
                self.update_gbest()

            self.pbest_replacement_batchcounts[obs_interval_idx] = self.pbest_replacement_counts
            self.pbest_replacement_counts = np.zeros(self.swarm_size)

    def diversely_optimize(self):
        stagnation_counter = 0
        previous_gbest_val = np.inf

        for obs_interval_idx in range(self.num_swarm_obs_intervals):
            for _ in range(self.swarm_obs_interval_length):
                self.update_velocities(self.gbest_pos)  # Use global leader particle position
                self.update_position()
                self.update_pbest()

                # Standard gbest update
                self.update_gbest()

                # Check for stagnation
                if self.gbest_val < previous_gbest_val:
                    previous_gbest_val = self.gbest_val
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # If the swarm has stagnated, consider selecting a diverse global best
                if stagnation_counter >= 10:  # Stagnation threshold, adjust as necessary
                    self.select_diverse_global_best()
                    stagnation_counter = 0  # Reset stagnation counter after diversification

            self.pbest_replacement_batchcounts[obs_interval_idx] = self.pbest_replacement_counts
            self.pbest_replacement_counts = np.zeros(self.swarm_size)





