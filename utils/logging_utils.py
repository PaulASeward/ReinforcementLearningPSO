import time
import numpy as np
import csv
import tensorflow as tf
from utils.plot_utils import plot_standard_results, plot_continuous_actions, plot_discrete_actions


class ResultsLogger:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.train_steps = config.train_steps
        self.num_episodes = config.num_episodes
        self.start_time = time.time()
        self.last_log_interval_time = self.start_time

        self.average_loss = []
        self.average_returns = []
        self.average_fitness = []

        # Initialize the results with to be stored
        self.swarm_episode_observations = np.empty((0, 20), dtype=object)  # NumPy array for better handling
        self.eval_interval_count = 0

        self.save_config()

    def plot_results(self):
        raise NotImplementedError

    def save_actions(self, actions_row):
        raise NotImplementedError

    def save_config(self):
        with open(self.config.experiment_config_path, 'w') as f:
            for key, value in vars(self.config).items():
                f.write(f"{key}: {value}\n")

    def save_step_results(self, epsilon, rewards, train_loss=None, swarm_observations_dicts=None):
        if train_loss is None:
            train_loss = 0.0
        else:
            train_loss = np.mean(train_loss)

        # for ep_dict in swarm_observations_dicts:
        #     print(ep_dict)

        cumulative_episode_reward = np.sum(rewards)
        fitness = self.config.fDeltas[self.config.func_num - 1] - cumulative_episode_reward

        self._save_to_csv(rewards, self.config.action_values_path)
        swarm_observations = np.array([swarm_observations_dicts], dtype=object)  # Convert dict to array
        self.swarm_episode_observations = np.vstack([self.swarm_episode_observations, swarm_observations])

        self._save_to_csv([epsilon], self.config.epsilon_values_path)  # TODO: remove
        self._save_to_csv([self.step, epsilon, cumulative_episode_reward, fitness, train_loss], self.config.training_step_results_path)

        print(f"Step #{self.step} Reward:{cumulative_episode_reward} Current Epsilon: {epsilon}")
        tf.summary.scalar("episode_reward", cumulative_episode_reward, step=self.step-1)

    def write_actions_at_eval_interval_to_csv(self):
        raise NotImplementedError

    def save_log_statements(self, step, actions, rewards, train_loss=None, epsilon=None, swarm_observations=None):
        self.step = step
        self.save_actions(actions)
        self.save_step_results(epsilon, rewards, train_loss, swarm_observations)

        if step % self.config.log_interval == 0:
            self.store_results_at_log_interval(train_loss)

        if step % self.config.eval_interval == 0:
            self.store_results_at_eval_interval()

    def print_execution_time(self):
        print(f"--- Execution took {(time.time() - self.start_time) / 3600} hours ---")

    def store_results_at_log_interval(self, train_loss=None):
        if train_loss is None:
            train_loss = 0.0
        else:
            train_loss = np.mean(train_loss)

        self.average_loss.append(train_loss)
        np.savetxt(self.config.loss_file, self.average_loss, delimiter=", ", fmt='% s')

        # Calculate average episode time
        current_time = time.time()
        log_interval_run_time = current_time - self.last_log_interval_time
        self.last_log_interval_time = current_time
        average_episode_time = log_interval_run_time / self.config.log_interval

        print(f"Step #{self.step} Loss:{train_loss} Average Episode Time: {average_episode_time / 60} minutes")
        # print('step = {0}: loss = {1}'.format(step, train_loss))

    def store_results_at_eval_interval(self):
        # Read the last eval_interval number of rows to calculate the total return per row. This can be used to then calculate the relative fitness and then compute averages.
        rewards = np.genfromtxt(self.config.action_values_path, delimiter=',')
        recent_rewards = rewards[-self.config.eval_interval:, :]
        reward_sums = np.sum(recent_rewards, axis=1)
        fitness = self.config.fDeltas[self.config.func_num - 1] - reward_sums

        avg_return = np.mean(reward_sums)  # Total return of all episodes for an iteration
        avg_fitness = np.mean(fitness)  # Furthest minimum value explored for an iteration

        self.average_returns.append(avg_return)
        self.average_fitness.append(avg_fitness)

        self.write_actions_at_eval_interval_to_csv()
        self.eval_interval_count += 1

        np.savetxt(self.config.average_returns_path, self.average_returns, delimiter=", ", fmt='% s')
        np.savetxt(self.config.fitness_path, self.average_fitness, delimiter=", ", fmt='% s')

        print('step = {0}: Average Return = {1} Average Fitness = {2}'.format(self.step, avg_return, avg_fitness))

    def _save_to_numpy(self, path, data):
        np.save(path, data)

    def _save_to_csv(self, data_row, path):
        with open(path, mode='a', newline='') as csv_file:
            csv.writer(csv_file).writerow(data_row)


class DiscreteActionsResultsLogger(ResultsLogger):
    def __init__(self, config):
        super().__init__(config)
        self.action_counts = np.zeros((self.config.num_eval_intervals, self.config.num_actions), dtype=np.int32)

    def save_actions(self, actions_row):
        for action in actions_row:
            self.action_counts[self.eval_interval_count, action] += 1

        self._save_to_csv(actions_row, self.config.action_counts_path)

    def write_actions_at_eval_interval_to_csv(self):
        self._save_to_csv(self.action_counts[self.eval_interval_count, :], self.config.interval_actions_counts_path)

    def plot_results(self):
        plot_standard_results(self.config)
        plot_discrete_actions(self.config)


class ContinuousActionsResultsLogger(ResultsLogger):
    def __init__(self, config):
        super().__init__(config)
        self.continuous_action_history = np.zeros((self.config.train_steps, self.config.num_episodes, self.config.action_dimensions), dtype=np.float32)

    def save_actions(self, actions_row):
        for episode_idx, action in enumerate(actions_row):
            self.continuous_action_history[self.step-1, episode_idx] = action

        # Save to disk periodically
        if self.step % self.config.log_interval == 0 or self.step == self.config.train_steps:
            self._save_to_numpy(self.config.continuous_action_history_path, self.continuous_action_history)

    def write_actions_at_eval_interval_to_csv(self):
        pass

    def plot_results(self):
        plot_standard_results(self.config)
        plot_continuous_actions(self.config)


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
