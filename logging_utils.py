import time
from abc import ABC, abstractmethod

import numpy as np
import os
import csv
from plot_utils import plot_results_over_iterations, plot_actions_over_iteration_intervals, plot_actions_from_env


class ComputeReturnStrategy(ABC):
    @abstractmethod
    def compute_average_return(self, env, model, num_episodes):
        pass


class ComputeDqnReturn(ComputeReturnStrategy):
    def compute_average_return(self, env, model, num_episodes=4):
        total_return = 0.0
        total_fitness = 0.0
        for _ in range(num_episodes):

            time_step = env.reset()
            observation = time_step.observation
            episode_return = 0.0

            terminal = False
            while not terminal:  # This is repeats logic of for _ in range, so we are taking new and separate 100 episodes.
                action = model.get_action(observation)
                step_type, reward, discount, observation = env.step(action)

                episode_return += reward.numpy()[0]
                terminal = bool(1 - discount)

            total_return += episode_return
            total_fitness += env.pyenv.envs[0]._best_fitness

        avg_return = total_return / num_episodes
        avg_fitness = total_fitness / num_episodes

        return avg_return, avg_fitness


class ComputeDrqnReturn(ComputeReturnStrategy):
    def compute_average_return(self, env, model, num_episodes=4):
        total_return = 0.0
        total_fitness = 0.0

        # Temporary Solution
        def update_states(states, next_state):
            states = np.roll(states, -1, axis=0)
            states[-1] = next_state
            return states

        for _ in range(num_episodes):
            states = np.zeros([10, 150])  # Make Dynamic

            current_state = env.reset()
            states = update_states(states, current_state.observation)

            episode_return = 0.0

            terminal = False
            while not terminal:  # This is repeats logic of for _ in range, so we are taking new and separate 100 episodes.
                action = model.get_action(states)
                step_type, reward, discount, next_state = env.step(action)

                episode_return += reward.numpy()[0]
                terminal = bool(1 - discount)

                states = update_states(states, next_state)

            total_return += episode_return
            total_fitness += env.pyenv.envs[0]._best_fitness

        avg_return = total_return / num_episodes
        avg_fitness = total_fitness / num_episodes

        return avg_return, avg_fitness


class ResultsLogger:
    def __init__(self, config, env, model, logging_strategy: ComputeReturnStrategy):
        self.config = config
        self.start_time = time.time()
        self.env = env
        self.model = model
        self.logging_strategy: ComputeReturnStrategy = logging_strategy

        self.loss = []
        self.returns = []
        self.fitness = []

        self.action_counts = np.zeros((self.config.num_eval_intervals, 5), dtype=np.int32)
        self.eval_interval_count = 0

    def save_log_statements(self, step, actions, train_loss=None):
        for action in actions:
            self.action_counts[self.eval_interval_count, action] += 1

        if step % self.config.log_interval == 0:
            if train_loss is None:
                train_loss = 0.0
            print('step = {0}: loss = {1}'.format(step, train_loss))
            self.loss.append(train_loss)
            np.savetxt(self.config.loss_file, self.loss, delimiter=", ", fmt='% s')

        if step % self.config.eval_interval == 0:
            avg_return, avg_fitness = self.logging_strategy.compute_average_return(self.env, self.model, 4)
            # # Mock Data:
            # avg_return, avg_fitness = 2.6, 3.2

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

    def plot_log_statements(self):
        plot_results_over_iterations(self.config.average_returns_plot_path, 'Average Return', 'Iteration', self.config.iterations, self.returns)
        plot_results_over_iterations(self.config.fitness_plot_path, 'Average Fitness', 'Iteration', self.config.iterations, self.fitness)
        plot_actions_over_iteration_intervals(self.config.interval_actions_plot_path, 'Iteration Intervals', 'Action Count', 'Action Distribution Over Iteration Intervals', self.config.iteration_intervals, self.config.label_iterations_intervals, self.action_counts)
        plot_actions_from_env(self.config.env_action_counts, self.config.env_action_values, 9)
        print(f"--- Execution took {(time.time() - self.start_time) / 3600} hours ---")
    def get_returns(self):
        return self.returns

    def get_fitness(self):
        return self.fitness

    def get_loss(self):
        return self.loss

    def get_action_counts(self):
        return self.action_counts
