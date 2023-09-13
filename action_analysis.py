import numpy as np
from matplotlib import pyplot as plt

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