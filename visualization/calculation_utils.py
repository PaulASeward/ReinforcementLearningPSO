import environment.functions as functions
import numpy as np
import matplotlib.colors as mcolors  # Import matplotlib colors
import math
import plotly.express as px


class EvaluationFunction:
    def __init__(self, fun_num, dim=2):
        self.obj_f = functions.CEC_functions(dim=dim, fun_num=fun_num)
        self.data_dir = f'data/f{fun_num}/'

    def eval(self, X):
        return self.obj_f.Y_matrix(np.array(X).astype(float))

    def get_swarm_data(self, step=250, episode=0):
        data_dir = self.data_dir + f'locations_at_step_{step}/'

        positions = np.load(data_dir + 'swarm_locations.npy')  # Shape is  (episodes, time_steps, particles, dimensions)
        ep_positions = positions[episode]  # Shape is (time_steps, particles, dimensions)

        velocities = np.load(data_dir + 'swarm_velocities.npy')  # Shape is (time_steps, particles, dimensions)
        ep_velocities = velocities[episode]  # Shape is (time_steps, particles, dimensions)

        swarm_best_positions = np.load(data_dir + 'swarm_best_locations.npy')  # Shape is (time_steps, particles, dim)
        ep_swarm_best_positions = swarm_best_positions[episode]  # Shape is (time_steps, particles, dimensions)

        valuations = np.load(data_dir + 'swarm_evaluations.npy')  # Shape is (time_steps, particles)
        ep_valuations = valuations[episode]  # Shape is (time_steps, particles)

        meta_data = np.genfromtxt(data_dir + 'meta_data.csv', delimiter=',', dtype=None, names=True, encoding='utf-8')
        ep_meta_data = meta_data[episode]  # Meta data for the first episode

        return ep_positions, ep_velocities, ep_swarm_best_positions, ep_valuations, ep_meta_data


def calculate_current_min_explored(positions, valuations):
    # Calculate the minimum value explored by the swarm and store the position and value at each time step.
    X1_min = None
    X2_min = None
    min_valuation = None

    min_explored = []

    for t in range(len(valuations)):
        current_positions = positions[t]
        current_valuations = valuations[t]
        min_valuation_t = np.min(current_valuations)

        if min_valuation is None or min_valuation_t < min_valuation:
            min_valuation = min_valuation_t
            min_valuation_index = np.argmin(current_valuations)
            X1_min = current_positions[min_valuation_index, 0]
            X2_min = current_positions[min_valuation_index, 1]

        min_explored.append([X1_min, X2_min, min_valuation])

    return min_explored

def get_distinct_colors(n):
    """Generate n distinct colors, using a cycling method if n exceeds the base palette size."""
    base_colors = px.colors.qualitative.Dark24  # This is a palette of dark colors
    if n <= len(base_colors):
        return base_colors[:n]
    else:
        # Extend the color palette by repeating and modifying slightly
        colors = []
        cycle_count = int(np.ceil(n / len(base_colors)))
        for i in range(cycle_count):
            for color in base_colors:
                modified_color = lighten_color(color, amount=0.1 * i)
                colors.append(modified_color)
                if len(colors) == n:
                    return colors
    return colors

def lighten_color(color, amount=0.5):
    """Lighten color by a given amount. Amount > 0 to lighten, < 0 to darken."""
    try:
        c = mcolors.to_rgb(color)
        c = mcolors.rgb_to_hsv(c)
        c = (c[0], c[1], max(0, min(1, c[2] * (1 + amount))))
        c = mcolors.hsv_to_rgb(c)
        return mcolors.to_hex(c)
    except:
        print('Error: Invalid color: ', color)
        return color


def generate_nonlinear_marks(shift, z_range):
    nonlinear_marks = {}
    exponential_ceiling = math.ceil(math.log10(z_range))
    linear_steps = list(range(0, (exponential_ceiling+1)*100, 100))  # Linear steps for the slider ex) [0,100,200,300,400]

    for i, val in enumerate(linear_steps):
        nonlinear_label = (10 ** i) - shift
        nonlinear_marks[val] = f'{nonlinear_label}'

    return nonlinear_marks


def linear_to_nonlinear_value(linear_value, shift):
    # Convert linear value to the original scale
    exponent = int(linear_value) / 100
    nonlinear_value = (10 ** exponent) - shift
    return int(nonlinear_value)
