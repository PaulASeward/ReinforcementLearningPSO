import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import os
import tensorflow as tf

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


def plot_actions_from_env(input_file_actions, input_file_values, num_intervals):
    output_file_name = os.path.splitext(input_file_actions)[0] + '.png'
    action_counts = np.genfromtxt(input_file_actions, delimiter=',')
    action_values = np.genfromtxt(input_file_values, delimiter=',')

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

    # Calculate the min and max values for the line graph
    min_line_value = np.inf  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = action_values[start_idx:end_idx]
        average_value_per_episode = np.mean(interval_values, axis=0)
        min_line_value = min(min_line_value, np.min(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid

        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_data = action_counts[start_idx:end_idx]
        interval_values = action_values[start_idx:end_idx]
        average_value_per_episode = np.mean(interval_values, axis=0)

        bottom = np.zeros(num_episodes)
        for action_num in range(num_actions):
            action_occurrences = [np.count_nonzero(interval_data[:, episode] == action_num) for episode in range(num_episodes)]
            ax.bar(x_values, action_occurrences, bottom=bottom, label=action_names[action_num])
            bottom += action_occurrences

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o')
        ax2.set_ylabel("Average Value of Error Per Episode")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Action Count")
        ax.set_title(f'Action Counts - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)