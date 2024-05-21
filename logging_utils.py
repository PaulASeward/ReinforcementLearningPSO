import time
from abc import ABC, abstractmethod

import csv
from plot_utils import *
import tensorflow as tf
from agents.utils.policy import *


class ComputeReturnStrategy(ABC):
    def __init__(self):
        self.policy = GreedyPolicy()  # Policy for Evaluating the Average Return of the Current Model

    @abstractmethod
    def compute_average_return(self, env, model, num_returns_to_average):
        pass


class ComputeDqnReturn(ComputeReturnStrategy):
    def compute_average_return(self, env, model, num_returns_to_average=4):
        total_return = 0.0
        total_fitness = 0.0

        for _ in range(num_returns_to_average):
            time_step = env.reset()
            observation = time_step.observation
            episode_return = 0.0

            terminal = False
            while not terminal:
                # action = model.get_action_q_values(observation)
                q_values = model.get_action_q_values(observation)
                action = self.policy.select_action(q_values)
                step_type, reward, discount, observation = env.step(action)

                episode_return += reward.numpy()[0]
                terminal = bool(1 - discount)

            total_return += episode_return
            total_fitness += env.pyenv.envs[0]._best_fitness

        avg_return = total_return / num_returns_to_average
        avg_fitness = total_fitness / num_returns_to_average

        return avg_return, avg_fitness


class ComputeDrqnReturn(ComputeReturnStrategy):
    def compute_average_return(self, env, model, num_returns_to_average=4):
        total_return = 0.0
        total_fitness = 0.0

        def update_states(states, next_state):
            states = np.roll(states, -1, axis=0)
            states[-1] = next_state
            return states

        for _ in range(num_returns_to_average):
            states = np.zeros([model.config.trace_length, model.config.observation_length])  # Make Dynamic

            current_state = env.reset()
            states = update_states(states, current_state.observation)

            episode_return = 0.0

            terminal = False
            while not terminal:  # This is repeats logic of for _ in range, so we are taking new and separate X episodes.
                q_values = model.get_action_q_values(np.reshape(states, [1, model.config.trace_length, model.config.observation_length]))
                action = self.policy.select_action(q_values)
                step_type, reward, discount, next_state = env.step(action)

                episode_return += reward.numpy()[0]
                terminal = bool(1 - discount)

                states = update_states(states, next_state)

            total_return += episode_return
            total_fitness += env.pyenv.envs[0]._best_fitness

        avg_return = total_return / num_returns_to_average
        avg_fitness = total_fitness / num_returns_to_average

        return avg_return, avg_fitness


class ResultsLogger:
    def __init__(self, config, env, raw_env, model, logging_strategy: ComputeReturnStrategy):
        self.config = config
        self.start_time = time.time()
        self.env = env
        self.raw_env = raw_env
        self.model = model
        self.logging_strategy: ComputeReturnStrategy = logging_strategy

        self.loss = []
        self.returns = []
        self.fitness = []

        self.action_counts = np.zeros((self.config.num_eval_intervals, self.config.num_actions), dtype=np.int32)
        self.eval_interval_count = 0

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

            # Save the locations and valuations to each directory
            if self.config.track_locations:
                locations_new_dir = os.path.join(self.config.swarm_locations_dir, f"locations_at_step_{step}")
                os.makedirs(locations_new_dir, exist_ok=True)

                env_swarm_locations_path = os.path.join(locations_new_dir, self.config.env_swarm_locations_name)
                emv_swarm_velocities_path = os.path.join(locations_new_dir, self.config.env_swarm_velocities_name)
                env_swarm_best_locations_path = os.path.join(locations_new_dir, self.config.env_swarm_best_locations_name)
                env_swarm_evaluations_path = os.path.join(locations_new_dir, self.config.env_swarm_evaluations_name)
                env_meta_data_path = os.path.join(locations_new_dir, self.config.env_meta_data_name)
                self.raw_env.store_locations_and_valuations(True, env_swarm_locations_path, emv_swarm_velocities_path, env_swarm_best_locations_path, env_swarm_evaluations_path, env_meta_data_path)

            avg_return, avg_fitness = self.logging_strategy.compute_average_return(self.env, self.model, 4)

            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            self.returns.append(float(avg_return))
            self.fitness.append(avg_fitness)

            with open(self.config.interval_actions_counts_path, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(self.action_counts[self.eval_interval_count, :])

            self.eval_interval_count += 1

            # saving the results into a TEXT file
            np.savetxt(self.config.average_returns_path, self.returns, delimiter=", ", fmt='% s')
            np.savetxt(self.config.fitness_path, self.fitness, delimiter=", ", fmt='% s')

    def store_episode_actions_to_csv(self, actions_row, values_row):
        with open(self.config.action_counts_path, mode='a', newline='') as csv_file:
            csv.writer(csv_file).writerow(actions_row)

        with open(self.config.action_values_path, mode='a', newline='') as csv_file:
            csv.writer(csv_file).writerow(values_row)

    def plot_log_statements(self):
        plot_data_over_iterations(self.config.average_returns_path, 'Average Return', 'Iteration', self.config.eval_interval)
        plot_data_over_iterations(self.config.fitness_path, 'Average Fitness', 'Iteration', self.config.eval_interval)
        plot_data_over_iterations(self.config.loss_file, 'Average Loss', 'Iteration', self.config.log_interval)
        plot_actions_over_iteration_intervals(self.config.interval_actions_counts_path, 'Iteration Intervals', 'Action Count', 'Action Distribution Over Iteration Intervals', self.config.iteration_intervals, self.config.label_iterations_intervals, self.config.action_names)
        plot_actions_with_values_over_iteration_intervals(self.config.action_counts_path, self.config.action_values_path, num_actions=self.config.num_actions, action_names=self.config.action_names)
        print(f"--- Execution took {(time.time() - self.start_time) / 3600} hours ---")

    def get_returns(self):
        return self.returns

    def get_fitness(self):
        return self.fitness

    def get_loss(self):
        return self.loss

    def get_action_counts(self):
        return self.action_counts


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
