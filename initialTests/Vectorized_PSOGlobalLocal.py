import random
import numpy as np
import math
import functions
import csv


class PSOVectorSwarmGlobalLocal:

    def __init__(self, objective_function, fun_num, dimension, evals_per_dim, swarm_size=50, RangeF=100, global_topology=True):
        # Store Function Evaluator and current evaluation column vector for each particle's best.
        self.w = 0.729844  # Inertia weight to prevent velocities becoming too large
        self.c1 = 2.05 * self.w  # Social component Learning Factor
        self.c2 = 2.05 * self.w  # Cognitive component Learning Factor

        self.function = objective_function
        self.fun_num = fun_num
        self.dimension = dimension
        self.evals_per_dim = evals_per_dim
        self.iterations = math.floor(evals_per_dim * dimension / swarm_size)  # maxevals = evalspPerDim * dimension
        self.swarm_size = swarm_size
        self.global_topology = global_topology


        # Initialize 3 matrices for Current Position, Velocity, and Position of particles' best solution
        self.X = np.random.uniform(low=-1 * RangeF, high=RangeF, size=(swarm_size, dimension))
        self.V = np.full((swarm_size, dimension), 0)
        self.P = self.X

        # Set Constraints for clamping position and limiting velocity
        self.Vmin, self.Vmax = -1 * RangeF, RangeF
        self.Xmin, self.Xmax = -1 * RangeF, RangeF

        # Record the initialized particle and global best solutions
        self.val = self.eval(self.P)  # Vector of particle's current value of its position based on Function Eval
        self.pbest_val = self.val  # Vector of particle's best value of a visited position.
        self.gbest_pos = self.P[np.argmin(self.pbest_val)]  # Vector of globally best visited position.
        self.gbest_val = np.min(self.pbest_val)  # Single value of the global best position

        # Local Best Initial Solutions
        self.Lbest_pos = self.P  # Matrix of best locally visited positions for each particle
        self.lbest_val = self.pbest_val  # Vector of each particle's value of best locally visited position

    def eval(self, X):
        arr_X = np.array(X).astype(float)
        return self.function.Y_matrix(arr_X, self.fun_num)

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
        # Add velocity to current position
        self.X = self.X + self.V

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

        # Use function evaluation for each particle (vector) in Swarm to provide
        # value for each position in X.
        self.val = self.eval(self.X)

    def update_pbest(self):
        # Update each Particle's best position for each particle dimension
        improved_particles = self.val < self.pbest_val
        self.P = np.where(improved_particles[:, np.newaxis], self.X, self.P)
        self.pbest_val = np.where(improved_particles, self.val, self.pbest_val)

    def update_gbest(self):
        self.gbest_pos = self.P[np.argmin(self.pbest_val)]
        self.gbest_val = np.min(self.pbest_val)

    def update_lbest(self):
        # Update the local best particle position and value for each particle.
        # This will use a Ring topology approach for the previous and subsequent neighbouring particles.

        # VECTORIZED LOCAL APPROACH
        # Find the indices of the previous, current, and next particles (wrapping around if necessary)
        prev_indices = np.roll(np.arange(self.swarm_size), 1)
        next_indices = np.roll(np.arange(self.swarm_size), -1)

        # Get the personal best values of the previous, current, and next particles
        prev_pbest_val = self.pbest_val[prev_indices]
        curr_pbest_val = self.pbest_val
        next_pbest_val = self.pbest_val[next_indices]

        # Find the index and value of the minimum personal best value among previous, current, and next particles
        min_pbest_index = np.argmin([prev_pbest_val, curr_pbest_val, next_pbest_val], axis=0)

        # Update the local best positions for the particles with a new minimum value
        self.Lbest_pos[min_pbest_index == 0] = np.roll(self.P, 1, axis=0)[min_pbest_index == 0]
        self.Lbest_pos[min_pbest_index == 2] = np.roll(self.P, -1, axis=0)[min_pbest_index == 2]

        # Update the local best values
        self.lbest_val = np.minimum.reduce([prev_pbest_val, curr_pbest_val, next_pbest_val])

        return

    def optimize(self):

        if self.global_topology:
            for _ in range(self.iterations):
                self.update_velocities(self.gbest_pos)  # Input global leader particle position
                self.update_position()
                self.update_pbest()
                self.update_gbest()
            # Return the global best value and global best position respectively
            return self.gbest_val, self.gbest_pos

        else: # Local ring topology with adjacent neighbor comparisons
            self.update_lbest()
            for _ in range(self.iterations):
                self.update_velocities(self.Lbest_pos)  # Input local leader particle position
                self.update_position()
                self.update_pbest()
                self.update_lbest()
            # return the global best value and global best position respectively
            return np.min(self.pbest_val), self.P[np.argmin(self.pbest_val)]

