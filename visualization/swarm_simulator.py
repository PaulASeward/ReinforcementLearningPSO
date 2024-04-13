import environment.functions as functions
import numpy as np
import os
from surface import Surface
from calculation_utils import get_distinct_colors, lighten_color


class SwarmSimulator:
    def __init__(self, fun_num, dim=2):
        self.obj_f = functions.CEC_functions(dim=dim, fun_num=fun_num)
        self.data_dir = f'data/f{fun_num}/'
        self.step_data_dir = None

        self.positions = None
        self.ep_positions = None
        self.velocities = None
        self.ep_velocities = None
        self.swarm_best_positions = None
        self.ep_swarm_best_positions = None
        self.valuations = None
        self.ep_valuations = None
        self.meta_data = None
        self.ep_meta_data = None

        self.num_particles = None
        self.dark_colors = None
        self.light_colors = None

        self.min_explored = None
        self.surface = Surface(eval_function=self.eval)

        self.load_completed_swarm_data(step=250, episode=0)

    def eval(self, X):
        return self.obj_f.Y_matrix(np.array(X).astype(float))

    def create_simulated_swarm_data(self, episodes=20):
        pass

    def load_particle_data_for_episode(self):
        self.calculate_current_min_explored()
        self.num_particles = self.ep_positions.shape[1]

        self.dark_colors = get_distinct_colors(self.num_particles)
        self.light_colors = [lighten_color(color, amount=0.5) for color in self.dark_colors]

    def get_available_episodes(self):
        data_directories = os.listdir(self.data_dir)
        episodes = [int(directory.split('_')[-1]) for directory in data_directories]
        return episodes

    def load_completed_swarm_data(self, step=250, episode=0):
        # There is typically 20 episodes with 100 timesteps, 10 particles, and 2 dimensions
        self.step_data_dir = self.data_dir + f'locations_at_step_{step}/'

        self.positions = np.load(self.step_data_dir + 'swarm_locations.npy')  # Shape is  (episodes, time_steps, particles, dimensions)
        self.ep_positions = self.positions[episode]  # Shape is (time_steps, particles, dimensions)

        self.velocities = np.load(self.step_data_dir + 'swarm_velocities.npy')  # Shape is (time_steps, particles, dimensions)
        self.ep_velocities = self.velocities[episode]  # Shape is (time_steps, particles, dimensions)

        self.swarm_best_positions = np.load(self.step_data_dir + 'swarm_best_locations.npy')  # Shape is (time_steps, particles, dim)
        self.ep_swarm_best_positions = self.swarm_best_positions[episode]  # Shape is (time_steps, particles, dimensions)

        self.valuations = np.load(self.step_data_dir + 'swarm_evaluations.npy')  # Shape is (time_steps, particles)
        self.ep_valuations = self.valuations[episode]  # Shape is (time_steps, particles)

        self.meta_data = np.genfromtxt(self.step_data_dir + 'meta_data.csv', delimiter=',', dtype=None, names=True, encoding='utf-8')
        self.ep_meta_data = self.meta_data[episode]  # Meta data for the first episode

        self.load_particle_data_for_episode()

        return self.ep_positions, self.ep_velocities, self.ep_swarm_best_positions, self.ep_valuations, self.ep_meta_data

    def calculate_current_min_explored(self):
        # Calculate the minimum value explored by the swarm and store the position and value at each time step.
        X1_min = None
        X2_min = None
        min_valuation = None

        min_explored = []

        for t in range(len(self.ep_valuations)):
            current_positions = self.ep_positions[t]
            current_valuations = self.ep_valuations[t]
            min_valuation_t = np.min(current_valuations)

            if min_valuation is None or min_valuation_t < min_valuation:
                min_valuation = min_valuation_t
                min_valuation_index = np.argmin(current_valuations)
                X1_min = current_positions[min_valuation_index, 0]
                X2_min = current_positions[min_valuation_index, 1]

            min_explored.append([X1_min, X2_min, min_valuation])

        self.min_explored = min_explored
        return min_explored
