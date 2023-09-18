import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import os

action_names = ['Do Nothing', 'Reset Slower Half', 'Encourage Social Learning', 'Discourage Social Learning', 'Reset All']


def plot_results_over_iterations(file_name, y_label, x_label, iterations, y_data):
    plt.plot(iterations, y_data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(file_name, dpi='figure', format="png", metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    plt.close()


def plot_actions_over_iteration_intervals(file_name, x_label, y_label, title, iteration_intervals, label_iteration_intervals, action_counts):
    # Create a bar plot for each action
    bar_width = 0.8 * (iteration_intervals[1] - iteration_intervals[0])
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(iteration_intervals))

    for action in range(5):
        plt.bar(iteration_intervals, action_counts[:, action], bottom=bottom, width=bar_width, label=action_names[action])
        bottom += action_counts[:, action]
    # Set x-axis ticks and labels with rotated tick labels
    plt.xticks(label_iteration_intervals, labels=[str(i) for i in label_iteration_intervals], rotation=45, ha="right")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_actions_from_env(input_file_name, num_intervals):
    output_file_name = os.path.splitext(input_file_name)[0] + '.png'
    action_counts = np.genfromtxt(input_file_name, delimiter=',')
    num_episodes = 10
    num_actions = 5
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=f'C{i}') for i in range(num_actions)]
    fig.legend(legend_handles, action_names, loc='upper right', title="Actions")

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid

        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_data = action_counts[start_idx:end_idx]

        bottom = np.zeros(num_episodes)
        for action_num in range(num_actions):
            action_occurrences = [np.count_nonzero(interval_data[:, episode] == action_num) for episode in
                                  range(num_episodes)]
            ax.bar(x_values, action_occurrences, bottom=bottom, label=action_names[action_num])
            bottom += action_occurrences

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Action Count")
        ax.set_title(f'Action Counts - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()
