import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import os


def plot_standard_results(config):
    plot_data_over_iterations(config.average_returns_path, 'Average Return', 'Iteration', config.eval_interval)
    plot_data_over_iterations(config.fitness_path, 'Average Fitness', 'Iteration', config.eval_interval)
    plot_data_over_iterations(config.loss_file, 'Average Loss', 'Iteration', config.log_interval)
    plot_two_datasets_over_iterations(config.average_returns_path, 'Average Return', config.epsilon_values_path, 'Epsilon Of Policy', 'Iteration', config.eval_interval)
    plot_two_datasets_over_iterations(config.fitness_path, 'Average Fitness', config.epsilon_values_path, 'Epsilon Of Policy', 'Iteration', config.eval_interval)


def plot_discrete_actions(config):
    plot_actions_over_iteration_intervals(config.interval_actions_counts_path, config.fitness_path,
                                          'Iteration Intervals', 'Action Count',
                                          'Action Distribution Over Iteration Intervals',
                                          config.iteration_intervals,
                                          config.label_iterations_intervals,
                                          config.actions_descriptions)
    plot_actions_with_values_over_iteration_intervals(config.action_counts_path,
                                                      config.action_values_path,
                                                      standard_pso_values_path=config.standard_pso_path,
                                                      function_min_value=config.fDeltas[
                                                          config.func_num - 1],
                                                      num_actions=config.num_actions,
                                                      action_names=config.actions_descriptions)


def plot_continuous_actions(config):
    plot_data_over_iterations(config.actor_loss_file, 'Average Actor Loss', 'Iteration', config.log_interval)
    plot_data_over_iterations(config.critic_loss_file, 'Average Critic Loss', 'Iteration', config.log_interval)


    if config.swarm_algorithm == "PMSO":
        plot_average_continuous_actions_for_multiple_swarms(config.continuous_action_history_path,
                                                            config.action_values_path,
                                                            standard_pso_values_path=config.standard_pso_path,
                                                            function_min_value=config.fDeltas[
                                                                config.func_num - 1],
                                                            action_dimensions=config.action_dimensions,
                                                            action_names=config.actions_descriptions,
                                                            action_offset=config.continuous_action_offset,
                                                            num_intervals=9)
    else:
        plot_average_continuous_actions_for_single_swarm(config.continuous_action_history_path,
                                                         config.action_values_path,
                                                         standard_pso_values_path=config.standard_pso_path,
                                                         function_min_value=config.fDeltas[
                                                             config.func_num - 1],
                                                         action_dimensions=config.action_dimensions,
                                                         action_names=config.actions_descriptions,
                                                         action_offset=config.continuous_action_offset,
                                                         num_intervals=15)

def plot_data_over_iterations(file_name, y_label, x_label, iteration_interval_scale):
    y_data = np.genfromtxt(file_name, delimiter=',')
    output_file_name = os.path.splitext(file_name)[0] + '_plot.png'
    iterations = range(0, len(y_data) * iteration_interval_scale, iteration_interval_scale)
    plt.plot(iterations, y_data)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(output_file_name, dpi='figure', format="png", metadata=None,
                bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto')
    plt.close()


def plot_two_datasets_over_iterations(file_name, y_label, file_name2, y_label2, x_label, iteration_interval_scale):
    y_data = np.genfromtxt(file_name, delimiter=',')
    y_data2 = np.genfromtxt(file_name2, delimiter=',')
    iterations_y_data = range(0, len(y_data) * iteration_interval_scale, iteration_interval_scale)
    iterations_y_data2 = range(0, len(y_data2))

    output_file_name = os.path.splitext(file_name)[0] + '_dual_plot.png'

    fig, ax1 = plt.subplots()

    # Plot the first dataset on the primary y-axis
    line1, = ax1.plot(iterations_y_data, y_data, label=y_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    # Create a second y-axis for the second dataset
    ax2 = ax1.twinx()
    line2, = ax2.plot(iterations_y_data2, y_data2, 'r-', label=y_label2)
    ax2.set_ylabel(y_label2)

    # Combine both lines into a single legend
    lines = [line1, line2]
    labels = [y_label, y_label2]
    ax1.legend(lines, labels, loc='upper left')

    fig.tight_layout()
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_actions_over_iteration_intervals(file_name, relative_fitness, x_label, y_label, title, iteration_intervals, label_iteration_intervals, action_names):
    action_counts = np.genfromtxt(file_name, delimiter=',')
    relative_fitness_values = np.genfromtxt(relative_fitness, delimiter=',')
    output_file_name = os.path.splitext(file_name)[0] + '_plot.png'
    num_actions = action_counts.shape[1]
    print("Number of Actions Being Plotted: ", num_actions)
    print("Number of Iteration Intervals: ", len(iteration_intervals))
    print("Number of Labels for Iteration Intervals: ", len(label_iteration_intervals))
    print("Number of Action Counts Rows: ", action_counts.shape[0])

    if len(iteration_intervals) != action_counts.shape[0]:
        print(f"Warning: Mismatch between iteration intervals ({len(iteration_intervals)}) and action counts rows ({action_counts.shape[0]}). Adjusting to the minimum size.")
        min_length = min(len(iteration_intervals), action_counts.shape[0])
        iteration_intervals = iteration_intervals[:min_length]
        action_counts = action_counts[:min_length, :]
        label_iteration_intervals = label_iteration_intervals[:min_length]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create a bar plot for each action
    bar_width = 0.8 * (iteration_intervals[1] - iteration_intervals[0])
    bottom = np.zeros(len(iteration_intervals))

    for action in range(num_actions):
        ax1.bar(iteration_intervals, action_counts[:, action], bottom=bottom, width=bar_width, label=action_names[action])
        bottom += action_counts[:, action]

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Action Count')
    ax1.set_title(title)

    # Create a second y-axis for the relative fitness values
    ax2 = ax1.twinx()
    ax2.plot(iteration_intervals, relative_fitness_values, label='Average Relative Fitness', color='black', marker='o')
    ax2.set_ylabel('Average Relative Fitness')

    # Set x-axis ticks and labels with rotated tick labels
    ax1.set_xticks(label_iteration_intervals)
    ax1.set_xticklabels([str(i) for i in label_iteration_intervals], rotation=45, ha="right", fontsize=8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_average_continuous_actions_for_single_swarm(continuous_action_history_path, actions_values_path, standard_pso_values_path, function_min_value, action_dimensions, action_names, action_offset, num_intervals=9):
    output_file_name = os.path.splitext(continuous_action_history_path)[0] + '_single_swarm.png'
    action_counts = np.load(continuous_action_history_path)
    standard_pso_results = np.genfromtxt(standard_pso_values_path, delimiter=',', skip_header=1)
    standard_pso_distance = abs(function_min_value - standard_pso_results[:, 1])
    action_values = np.genfromtxt(actions_values_path, delimiter=',')
    # Fill in the action values with zeros to match the length of the action counts
    action_values = np.pad(action_values, ((0, len(action_counts) - len(action_values)), (0, 0)), 'constant')
    cumulative_rewards = np.cumsum(action_values, axis=1)

    num_episodes = action_counts.shape[1]
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=f'C{i}') for i in range(action_dimensions)]
    fig.legend(legend_handles, action_names[:action_dimensions], loc='upper right', title="Actions")

    # Calculate the min and max values for the line graph
    min_line_value = 0  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):  # Calculate min and max values for the line graph
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length

        interval_data = action_counts[start_idx:end_idx]
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(standard_pso_distance))

        mean_action_counts = np.mean(interval_data, axis=0)
        std_action_counts = np.std(interval_data, axis=0)

        # Plot each action dimension with a shaded area for the std deviation
        for j in range(action_dimensions):
            mean_counts = mean_action_counts[:, j]
            mean_with_offset = mean_counts + action_offset[j]
            std_dev_counts = std_action_counts[:, j]
            ax.plot(x_values, mean_with_offset, color=f'C{j}', label=action_names[j])
            ax.fill_between(x_values, mean_with_offset - std_dev_counts, mean_with_offset + std_dev_counts,
                            color=f'C{j}', alpha=0.3)

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
        ax2.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
        ax2.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Continuous Action Average")
        ax.set_title(f'Continuous Action Averages - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

        if i == 0:  # Only add legend to the first subplot to avoid repetition
            ax2.legend(loc='upper left')

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_average_continuous_actions_for_multiple_swarms(continuous_action_history_path, actions_values_path, standard_pso_values_path, function_min_value, action_dimensions, action_names, action_offset, num_intervals=9):
    output_file_name = os.path.splitext(continuous_action_history_path)[0] + '_multiple_swarms.png'
    action_counts = np.load(continuous_action_history_path)

    standard_pso_results = np.genfromtxt(standard_pso_values_path, delimiter=',', skip_header=1)
    standard_pso_distance = abs(function_min_value - standard_pso_results[:, 1])
    action_values = np.genfromtxt(actions_values_path, delimiter=',')
    # Fill in the action values with zeros to match the length of the action counts
    action_values = np.pad(action_values, ((0, len(action_counts) - len(action_values)), (0, 0)), 'constant')
    cumulative_rewards = np.cumsum(action_values, axis=1)

    num_episodes = action_counts.shape[1]
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=f'C{i}') for i in range(action_dimensions)]
    fig.legend(legend_handles, action_names[:action_dimensions], loc='upper right', title="Actions")

    # Calculate the min and max values for the line graph
    min_line_value = 0  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):  # Calculate min and max values for the line graph
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length

        interval_data = action_counts[start_idx:end_idx]
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(standard_pso_distance))

        mean_action_counts = np.mean(interval_data, axis=0)
        std_action_counts = np.std(interval_data, axis=0)

        # Plot each action dimension with a shaded area for the std deviation
        for j in range(action_dimensions):
            mean_counts = mean_action_counts[:, j]
            mean_with_offset = mean_counts + action_offset[j]
            std_dev_counts = std_action_counts[:, j]
            ax.plot(x_values, mean_with_offset, color=f'C{j}', label=action_names[j])
            ax.fill_between(x_values, mean_with_offset - std_dev_counts, mean_with_offset + std_dev_counts, color=f'C{j}', alpha=0.3)

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
        ax2.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
        ax2.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Continuous Action Average")
        ax.set_title(f'Continuous Action Averages - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

        if i == 0:  # Only add legend to the first subplot to avoid repetition
            ax2.legend(loc='upper left')

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()


def plot_actions_with_values_over_iteration_intervals(actions_counts_path, actions_values_path, standard_pso_values_path, function_min_value, num_actions, action_names, num_intervals=9):
    output_file_name = os.path.splitext(actions_counts_path)[0] + '.png'
    action_counts = np.genfromtxt(actions_counts_path, delimiter=',')
    standard_pso_results = np.genfromtxt(standard_pso_values_path, delimiter=',', skip_header=1)
    standard_pso_distance = abs(function_min_value - standard_pso_results[:, 1])
    action_values = np.genfromtxt(actions_values_path, delimiter=',')
    cumulative_rewards = np.cumsum(action_values, axis=1)

    num_episodes = action_counts.shape[1]
    rows_per_interval = len(action_counts) // num_intervals
    x_values = range(1, num_episodes + 1)

    # Calculate the number of rows and columns for the grid of plots
    num_rows = num_intervals // 3
    num_cols = min(num_intervals, 3)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6 * num_rows))

    # Create a legend for the actions using action names
    legend_handles = [Patch(facecolor=f'C{i}') for i in range(num_actions)]
    fig.legend(legend_handles, action_names[:num_actions], loc='upper right', title="Actions")

    # Calculate the min and max values for the line graph
    min_line_value = 0  # Initialize with a high value
    max_line_value = -np.inf  # Initialize with a low value
    for i in range(num_intervals):  # Calculate min and max values for the line graph
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))

    for i in range(num_intervals):
        row = i // 3
        col = i % 3
        ax = axes[row, col]  # Access the appropriate subplot in the grid
        start_idx = i * rows_per_interval
        end_idx = (i + 1) * rows_per_interval if i < num_intervals - 1 else len(action_counts)  # Final interval length

        interval_data = action_counts[start_idx:end_idx]
        interval_values = cumulative_rewards[start_idx:end_idx]
        average_value_per_episode = abs(np.mean(interval_values, axis=0))
        max_line_value = max(max_line_value, np.max(average_value_per_episode))
        max_line_value = max(max_line_value, np.max(standard_pso_distance))

        bottom = np.zeros(num_episodes)
        for action_num in range(num_actions):
            action_occurrences = [np.count_nonzero(interval_data[:, episode] == action_num) for episode in range(num_episodes)]
            ax.bar(x_values, action_occurrences, bottom=bottom, label=action_names[action_num])
            bottom += action_occurrences

        # Add line graph overlay
        ax2 = ax.twinx()
        ax2.plot(x_values, average_value_per_episode, color='black', marker='o', label='RL PSO')
        ax2.plot(x_values, standard_pso_distance, color='red', marker='o', label='Standard PSO')
        ax2.set_ylabel("Average Best Minimum Explored Relative to Function Minimum")
        ax2.set_ylim(min_line_value, max_line_value)

        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Action Count")
        ax.set_title(f'Action Counts - (Iterations {start_idx + 1} to {end_idx})')
        ax.set_xticks(x_values)

        if i == 0:  # Only add legend to the first subplot to avoid repetition
            ax2.legend(loc='upper left')

    # Adjust subplot layout and add single legend
    plt.tight_layout()

    # Save the single figure with subplots
    plt.savefig(output_file_name, dpi='figure', format="png", bbox_inches='tight')
    plt.close()
