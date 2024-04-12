import environment.functions as functions
import numpy as np


# Evaluation function (adjust as necessary)
obj_f = functions.CEC_functions(dim=2, fun_num=6)
def eval(X):
    return obj_f.Y_matrix(np.array(X).astype(float))


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


def generate_exponential_marks2(z_min, z_max):
    marks = {}
    num_marks = 10  # Target number of marks

    if z_min < 0:
        # Handle cases where z_min is negative and z_max is positive
        start_power = np.ceil(np.log2(abs(z_min)))
        end_power = np.ceil(np.log2(z_max))
    else:
        # Both z_min and z_max are non-negative
        start_power = np.ceil(np.log2(max(1, z_min)))  # Avoid log2(0)
        end_power = np.ceil(np.log2(z_max))

    # Determine the appropriate step size to achieve approximately 'num_marks' marks
    step_size = (end_power - start_power) / (num_marks - 1)

    # Generate exponential marks
    current_power = start_power
    while current_power <= end_power:
        current_value = 2 ** current_power
        marks[int(current_value)] = f'{int(current_value)}'
        current_power += step_size

    # Always include the maximum value if it's not already included
    max_mark = 2 ** end_power
    if int(max_mark) not in marks:
        marks[int(max_mark)] = f'{int(max_mark)}'

    # Include z_min if it is negative
    if z_min < 0:
        marks[int(z_min)] = f'{int(z_min)}'

    return marks

def generate_exponential_marks(z_min, z_max):
    marks = {}
    num_marks = 10  # Target number of marks

    # To avoid log(0) or log(negative) issues, shift the range if necessary
    epsilon = 1e-10
    shift = abs(min(z_min, 0)) + epsilon
    adj_z_min = z_min + shift
    adj_z_max = z_max + shift

    # Generate logarithmic steps
    log_min = np.log(adj_z_min)
    log_max = np.log(adj_z_max)
    step_size = (log_max - log_min) / (num_marks - 1)

    # Populate marks using the logarithmic scale converted back to the original scale
    for i in range(num_marks):
        log_value = log_min + i * step_size
        raw_value = np.exp(log_value)
        mark_value = int(raw_value - shift)  # Adjusting back to the original scale
        marks[mark_value] = f'{mark_value}'

    # Ensure the maximum value is included
    if z_max not in marks:
        marks[z_max] = f'{z_max}'

    print("Marks: ", marks)
    return marks


def generate_geometric_marks(z_min, z_max):
    marks = {}
    num_marks = 10  # Target number of marks
    factor = 5  # This is the multiplication factor for each step
    z_min, z_max = int(z_min), int(z_max)
    # Shift the range if necessary to avoid negative values
    shift = abs(min(z_min, 0)) if z_min <= 0 else - abs(max(z_min, 0))

    adj_z_min = z_min + shift + 1
    adj_z_max = z_max + shift

    # Initialize the first mark
    current_mark = int(adj_z_min)
    marks[current_mark - shift] = f'{current_mark - shift}'

    # Generate marks by multiplying the previous mark by the factor
    while current_mark * factor <= adj_z_max:
        current_mark *= factor
        mark_value = int(current_mark - shift)  # Adjusting back to the original scale
        marks[mark_value] = f'{mark_value}'

    # Ensure the last mark is z_max if it's not close to the last computed value
    if current_mark != adj_z_max:
        marks[adj_z_max-shift] = f'{adj_z_max-shift}'

    return marks