import numpy as np

class Particle:
    def __init__(self, x, v, p, P_val, current_valuation, velocity_magnitude, relative_fitness, average_batch_count):
        self.x = x
        self.v = v
        self.p = p
        self.P_val = P_val
        self.velocity_magnitude = velocity_magnitude
        self.current_valuation = current_valuation
        self.relative_fitness = relative_fitness
        self.average_batch_count = average_batch_count




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

        # Set Constraints for clamping position and limiting velocity
        self.abs_max_velocity = self.rangeF
        self.abs_max_position = self.rangeF

        self.velocity_magnitudes = None
        self.relative_fitnesses = None
        self.average_batch_counts = None
        self.pbest_replacement_counts = None

        self.P_vals = None
        self.current_valuations = None
        self.gbest_val = None
        self.gbest_pos = None

        # Initialize the swarm's positions velocities and best solutions
        self._initialize()

    def reinitialize(self):
        self._initialize()

    def _initialize(self):
        # Initialize 3 matrices for Current Position, Velocity, and Position of particles' best solution
        self.X = np.random.uniform(low=-1 * self.rangeF, high=self.rangeF, size=(self.swarm_size, self.dimension))
        self.V = np.full((self.swarm_size, self.dimension), 0)
        self.P = self.X
        self.P_vals = None

        # Reset the adjustable parameters to starting values
        self.w = self.config.w
        self.c1 = self.config.c1
        self.c2 = self.config.c2
        self.gbest_replacement_threshold, self.pbest_replacement_threshold = self.config.replacement_threshold, self.config.replacement_threshold
        self.abs_max_velocity = self.rangeF

        self.initialize_stored_counts()

        # Record the initialized particle and global best solutions
        self.update_swarm_valuations_and_bests()

    def initialize_stored_counts(self):
        self.velocity_magnitudes = None
        self.relative_fitnesses = None

        self.current_valuations = None
        self.gbest_val = None
        self.gbest_pos = None

        # Static class variables to track P_vals replacements
        self.pbest_replacement_counts = np.zeros(self.swarm_size)
        self.pbest_replacement_batchcounts = np.zeros((self.num_swarm_obs_intervals, self.swarm_size))
        self.average_batch_counts = np.zeros(self.swarm_size)

    def update_swarm_valuations_and_bests(self):
        self.P_vals = self.eval(self.P)  # Vector of particle's current valuation of its best position based on Function Eval
        self.current_valuations = self.eval(self.P)  # Vector of particle's current value of its position based on Function Eval
        self.gbest_pos = self.P[np.argmin(self.P_vals)]  # Vector of globally best visited position.
        self.gbest_val = np.min(self.P_vals)  # Single valuation of the global best position

        self.update_relative_fitnesses()
        self.update_velocity_maginitude()

    def update_relative_fitnesses(self):
        self.relative_fitnesses = (self.P_vals - self.gbest_val) / np.abs(self.P_vals)

    def update_velocity_maginitude(self):
        self.velocity_magnitudes = np.linalg.norm(self.V, axis=1)

    def update_batch_counts(self):
        # Calculate the average batch count for each particle
        self.average_batch_counts = np.mean(self.pbest_replacement_batchcounts, axis=0)

    def get_observation(self):
        self.update_relative_fitnesses()
        self.update_velocity_maginitude()
        self.update_batch_counts()

        return np.concatenate([self.velocity_magnitudes, self.relative_fitnesses, self.average_batch_counts], axis=0)

    def get_current_best_fitness(self):
        return self.gbest_val
    
    def get_current_best_fitnesses(self):
        return self.P_vals

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
        self.current_valuations = self.eval(self.X)

    def update_pbest(self):
        improved_particles = self.current_valuations < self.P_vals  # Update each Particle's best position for each particle index
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.P_vals = np.where(improved_particles, self.current_valuations, self.P_vals)

        self.pbest_replacement_counts += improved_particles  # Update P_vals replacement counter

    def update_gbest(self):
        self.gbest_pos = self.P[np.argmin(self.P_vals)]
        self.gbest_val = np.min(self.P_vals)

    def optimize(self):
        for obs_interval_idx in range(self.num_swarm_obs_intervals):
            for i in range(self.swarm_obs_interval_length):
                self.update_velocities(self.gbest_pos)  # Input global leader particle position
                self.update_position()
                self.update_pbest()
                self.update_gbest()

            self.pbest_replacement_batchcounts[obs_interval_idx] = self.pbest_replacement_counts
            self.pbest_replacement_counts = np.zeros(self.swarm_size)

    def get_particle(self, particle_index: int):
        return Particle(self.X[particle_index],
                        self.V[particle_index],
                        self.P[particle_index],
                        self.P_vals[particle_index],
                        self.current_valuations[particle_index],
                        self.velocity_magnitudes[particle_index],
                        self.relative_fitnesses[particle_index],
                        self.average_batch_counts[particle_index])

    def add_particle(self, particle: Particle, particle_index: int):
        self.X[particle_index] = particle.x
        self.V[particle_index] = particle.v
        self.P[particle_index] = particle.p
        self.P_vals[particle_index] = particle.P_val
        self.current_valuations[particle_index] = particle.current_valuation
        self.velocity_magnitudes[particle_index] = particle.velocity_magnitude
        self.relative_fitnesses[particle_index] = particle.relative_fitness
        self.average_batch_counts[particle_index] = particle.average_batch_count






