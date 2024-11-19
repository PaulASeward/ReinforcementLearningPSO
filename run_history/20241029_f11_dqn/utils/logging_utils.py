import time
import numpy as np
import csv
import tensorflow as tf


class ResultsLogger:
    def __init__(self, config):
        self.config = config
        self.start_time = time.time()

        self.loss = []
        self.returns = []
        self.fitness = []

        self.action_counts = np.zeros((self.config.num_eval_intervals, self.config.num_actions), dtype=np.int32)
        self.eval_interval_count = 0

    def print_execution_time(self):
        print(f"--- Execution took {(time.time() - self.start_time) / 3600} hours ---")

    def save_log_statements(self, step, actions, rewards, train_loss=None):
        for action in actions:
            self.action_counts[self.eval_interval_count, action] += 1

        self.store_episode_actions_to_csv(actions, rewards)

        if step % self.config.log_interval == 0:
            if train_loss is None:
                train_loss = 0.0
            else:
                train_loss = np.mean(train_loss)
            print('step = {0}: loss = {1}'.format(step, train_loss))
            self.loss.append(train_loss)
            np.savetxt(self.config.loss_file, self.loss, delimiter=", ", fmt='% s')

        if step % self.config.eval_interval == 0:

            # Read the last eval_interval number of rows to calculate the total return per row. This can be used to then calculate the relative fitness and then compute averages.
            rewards = np.genfromtxt(self.config.action_values_path, delimiter=',')
            recent_rewards = rewards[-self.config.eval_interval:, :]
            reward_sums = np.sum(recent_rewards, axis=1)
            fitness = self.config.fDeltas[self.config.func_num - 1] - reward_sums

            avg_return = np.mean(reward_sums)  # Total return of all episodes for an iteration
            avg_fitness = np.mean(fitness)  # Furthest minimum value explored for an iteration

            print('step = {0}: Average Return = {1} Average Fitness = {2}'.format(step, avg_return, avg_fitness))
            self.returns.append(avg_return)
            self.fitness.append(avg_fitness)

            with open(self.config.interval_actions_counts_path, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(self.action_counts[self.eval_interval_count, :])

            self.eval_interval_count += 1

            np.savetxt(self.config.average_returns_path, self.returns, delimiter=", ", fmt='% s')
            np.savetxt(self.config.fitness_path, self.fitness, delimiter=", ", fmt='% s')

    def store_episode_actions_to_csv(self, actions_row, values_row):
        with open(self.config.action_counts_path, mode='a', newline='') as csv_file:
            csv.writer(csv_file).writerow(actions_row)

        with open(self.config.action_values_path, mode='a', newline='') as csv_file:
            csv.writer(csv_file).writerow(values_row)



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
