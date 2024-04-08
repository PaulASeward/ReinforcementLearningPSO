import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import environment.functions as functions

# Load your data
# There is typically 20 episodes with 100 timesteps, 10 particles, and 2 dimensions
# We will use the first episode for this example
positions = np.load('data/swarm_locations.npy')  # Shape is  (episodes, time_steps, particles, dimensions)
positions = positions[0]  # Shape is (time_steps, particles, dimensions)

# # We technically don't need this, since we can evaluate through the eval() below.
# valuations = np.load('data/swarm_evaluations.npy')  # Shape is (time_steps, particles)
# valuations = valuations[0]  # Shape is (time_steps, particles)

# Function

obj_f = functions.CEC_functions(dim=2, fun_num=6)
def eval(X):
    return obj_f.Y_matrix(np.array(X).astype(float))

# Create a meshgrid for the background
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = eval([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Function to update the plot for each frame
def update(num, positions, ax):
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    # Extract the positions for the current timestep
    current_positions = positions[num]
    xs = current_positions[:, 0]
    ys = current_positions[:, 1]
    zs = eval([xs, ys])
    ax.scatter(xs, ys, zs, color='r')  # Plot current particle positions
    ax.set_xlim([-5.12, 5.12])
    ax.set_ylim([-5.12, 5.12])
    ax.set_zlim([0, 80])

# Creating the Animation object
ani = animation.FuncAnimation(fig, update, frames=range(positions.shape[0]), fargs=(positions, ax))

plt.show()