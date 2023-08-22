import tensorflow as tf
import functionstf
class PSOtf:

    def __init__(self, objective_function, fun_num, dimension, observation_interval, swarm_size=50, RangeF=100, action=None):
        self.w = 0.729844  # Inertia weight to prevent velocities from becoming too large
        self.c1 = 2.05 * self.w  # Social component Learning Factor
        self.c2 = 2.05 * self.w  # Cognitive component Learning Factor

        self.function = objective_function
        self.fun_num = fun_num
        self.dimension = dimension
        self.iterations = observation_interval
        self.swarm_size = swarm_size
        self.rangeF = RangeF

        # Static class variable to track pbest_val replacements
        self.pbest_replacements_counter = tf.zeros(swarm_size, dtype=tf.float32)
        self.pbest_replacement_batchcounts = tf.zeros((self.iterations // 1000, self.swarm_size), dtype=tf.float32)

        # Initialize the swarm's positions velocities and best solutions
        self._initialize()
        self.update_relative_fitness()

        # Set Constraints for clamping position and limiting velocity
        self.Vmin, self.Vmax = -1 * RangeF, RangeF
        self.Xmin, self.Xmax = -1 * RangeF, RangeF

    def reinitialize(self):
        self._initialize()

    def _initialize(self):
        print('arrived to _initialize')
        # Initialize 3 tensors for Current Position, Velocity, and Position of particles' best solution
        self.X = tf.random.uniform(shape=(self.swarm_size, self.dimension), minval=-1 * self.rangeF, maxval=self.rangeF)
        self.V = tf.zeros(shape=(self.swarm_size, self.dimension), dtype=tf.float32)
        self.P = self.X

        # Record the initialized particle and global best solutions
        self.val = self.eval(self.P)  # Vector of particle's current value of its position based on Function Eval
        self.pbest_val = self.val  # Vector of particle's best value of a visited position.
        self.gbest_pos = self.P[tf.argmin(self.pbest_val)]  # Vector of globally best visited position.
        self.gbest_val = tf.reduce_min(self.pbest_val)  # Single value of the global best position

    def update_relative_fitness(self):
        self.relative_fitness = 1 - (self.pbest_val - self.gbest_val) / tf.abs(self.pbest_val)

    def get_observation(self):
        print('arrived to get_observation')
        self.update_relative_fitness()
        velocity_magnitudes = tf.norm(self.V, axis=1)
        return velocity_magnitudes, self.relative_fitness, self.pbest_replacement_batchcounts

    def get_current_best_fitness(self):
        return self.gbest_val

    def eval(self, X):
        # arr_X = tf.cast(X, dtype=tf.float32)
        # return self.function.Y_matrix(arr_X, self.fun_num)
        X_shift = 0.05 * (X - self.function.O) + 1

        tmp = X_shift ** 2 - tf.roll(X_shift, shift=-1, axis=0)
        tmp = 100 * tmp ** 2 + (X_shift - 1) ** 2
        Y = tf.reduce_sum(tmp ** 2 / 4000 - tf.cos(tmp) + 1, axis=1) + 500
        return Y


    def update_velocities(self, leader):
        # Generate social and cognitive components of velocity matrix update using tf.random
        social = self.c1 * tf.random.uniform(shape=(self.swarm_size, self.dimension)) * (leader - self.X)
        cognitive = self.c2 * tf.random.uniform(shape=(self.swarm_size, self.dimension)) * (self.P - self.X)

        # Update new velocity with old velocity*inertia plus component matrices
        self.V = self.w * self.V + social + cognitive

        # Maintain velocity constraints
        self.V = tf.clip_by_value(self.V, self.Vmin, self.Vmax)

    def update_position(self):
        # Add velocity to current position
        self.X = self.X + self.V

        # Clamp position inside boundary and reflect them in case they are out of the boundary
        # based on: S. Helwig, J. Branke, and S. Mostaghim, "Experimental Analysis of Bound Handling Techniques in Particle Swarm Optimization,"
        # IEEE TEC: 17(2), 2013, pp. 259-271

        # Identify positions outside the boundaries
        out_of_bounds = tf.logical_or(self.X < self.Xmin, self.X > self.Xmax)

        # Reflect the out-of-bounds positions and set their velocities to zero
        self.X = tf.where(out_of_bounds, tf.sign(self.X) * 2 * self.Xmax - self.X, self.X)
        self.V = tf.where(out_of_bounds, 0, self.V)

        # Update the positions and evaluate the new positions
        pbest_replacement_counts = tf.zeros((self.iterations // 1000, self.swarm_size), dtype=tf.float32)

        # Use function evaluation for each particle (vector) in Swarm to provide
        # value for each position in X.
        self.val = self.eval(self.X)

    def update_pbest(self):
        # Update each Particle's best position for each particle dimension
        improved_particles = self.val < self.pbest_val
        self.P = tf.where(improved_particles[:, tf.newaxis], self.X, self.P)
        self.pbest_val = tf.where(improved_particles, self.val, self.pbest_val)

        # Update pbest_val replacement counter
        self.pbest_replacements_counter += tf.cast(improved_particles, tf.float32)

    def update_gbest(self):

        self.gbest_pos = self.P[tf.argmin(self.pbest_val)]
        self.gbest_val = tf.reduce_min(self.pbest_val)

    def optimize(self):

        replacement_peak_counter = 0
        for i in range(self.iterations):
            self.update_velocities(self.gbest_pos)  # Input global leader particle position
            self.update_position()
            self.update_pbest()
            self.update_gbest()
            # print('completed iteration: ', i)

            # if (i + 1) % 1000 == 0:
            #     # Store pbest_replacements counts and reset the array
            #     indices = tf.convert_to_tensor([[replacement_peak_counter, i] for i in range(self.swarm_size)])
            #     updates = tf.expand_dims(self.pbest_replacements_counter, axis=0)
            #
            #     self.pbest_replacement_batchcounts = tf.tensor_scatter_nd_update(self.pbest_replacement_batchcounts,
            #                                                                      indices, updates)
            #
            #     # self.pbest_replacement_batchcounts[replacement_peak_counter] = self.pbest_replacements_counter
            #     self.pbest_replacements_counter = tf.zeros(self.swarm_size, dtype=tf.float32)
            #
            #     replacement_peak_counter += 1
