import numpy as np
from matplotlib import pyplot as plt
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
        plt.bar(iteration_intervals, action_counts[:, action], bottom=bottom, width=bar_width,
                label=action_names[action])
        bottom += action_counts[:, action]
    # Set x-axis ticks and labels with rotated tick labels
    plt.xticks(label_iteration_intervals, labels=[str(i) for i in label_iteration_intervals], rotation=45, ha="right")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_actions_from_env(output_file_name, input_file_name, x_label, y_label, title, num_intervals):
    action_counts = np.genfromtxt(input_file_name, delimiter=',')
    num_episodes = 10
    num_actions = 5
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Create a single large figure with subplots for each interval
    fig, axes = plt.subplots(nrows=num_intervals, figsize=(10, 6 * num_intervals))

    for i in range(num_intervals):
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_data = action_counts[start_idx:end_idx]

        for episode in range(num_episodes):
            episode_data = interval_data[:, episode]
            action_occurrences = [np.count_nonzero(episode_data == action) for action in range(1, num_actions + 1)]

            # Create a single bar for the episode in the subplot
            bottom = np.zeros(num_actions)
            for action_num, count in enumerate(action_occurrences):
                axes[i].bar(episode + 1, count, bottom=bottom[action_num], label=f'Action {action_num + 1}')
                bottom[action_num] += count

        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel(y_label)
        axes[i].set_title(f'{title} - Interval {i + 1}')
        axes[i].set_xticks(x_values)
        axes[i].set_xticklabels([f'Episode {i}' for i in x_values], rotation=45)
        axes[i].legend(title="Action")

    # Adjust subplot layout
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


# Example usage:
plot_actions_from_env("output.png", '/home/paul/UPEI/PSO_RL_gh/ReinforcementLearningPSO/episode_actions/function19_.csv', "Episode Number", "Action Count", "Action Counts per Interval", 10)

