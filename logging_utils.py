import time
import numpy as np
import os
import csv
from plot_utils import plot_results_over_iterations, plot_actions_over_iteration_intervals, plot_actions_from_env


def compute_avg_return(env, model, num_episodes=4):
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


def compute_avg_recurrent_return(env, model, num_episodes=4):
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
    def __init__(self, config, env, model, max_episodes=1000):
        self.config = config
        self.config.num_iterations = max_episodes
        self.start_time = time.time()
        self.env = env
        self.model = model

        self.loss = []
        self.returns = []
        self.fitness = []

        self.num_eval_intervals = self.config.num_iterations // self.config.eval_interval
        self.action_counts = np.zeros((self.num_eval_intervals, 5), dtype=np.int32)
        self.eval_interval_count = 0

        self.iterations = range(0, self.config.num_iterations, self.config.eval_interval)
        self.iteration_intervals = range(self.config.eval_interval, self.config.num_iterations + self.config.eval_interval, self.config.eval_interval)
        self.label_iterations_intervals = range(0, self.config.num_iterations + self.config.eval_interval, self.config.eval_interval*2)

        os.makedirs(self.config.results_dir, exist_ok=True)
        # file = open(self.config.loss_file, "r")
        # self.loss.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
        # self.loss = [item for sublist in self.loss for item in sublist]
        # file.close()
        #
        # file = open(self.config.results_file_reward, "r")
        # self.returns.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
        # self.returns = [item for sublist in self.returns for item in sublist]
        # file.close()
        #
        # file = open(self.config.results_file_fitness, "r")
        # self.fitness.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
        # self.fitness = [item for sublist in self.fitness for item in sublist]
        # file.close()

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
            avg_return, avg_fitness = compute_avg_recurrent_return(self.env, self.model)
            # # Mock Data:
            # avg_return, avg_fitness = 2.6, 3.2

            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            self.returns.append(float(avg_return))
            self.fitness.append(avg_fitness)

            with open(self.config.results_actions, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(self.action_counts[self.eval_interval_count, :])

            self.eval_interval_count += 1

            # saving the results into a TEXT file
            np.savetxt(self.config.results_file_reward, self.returns, delimiter=", ", fmt='% s')
            np.savetxt(self.config.results_file_fitness, self.fitness, delimiter=", ", fmt='% s')

    def plot_log_statements(self):
        plot_results_over_iterations(self.config.figure_file_rewards, 'Average Return', 'Iteration', self.iterations, self.returns)
        plot_results_over_iterations(self.config.figure_file_fitness, 'Average Fitness', 'Iteration', self.iterations, self.fitness)
        plot_actions_over_iteration_intervals(self.config.figure_file_action, 'Iteration Intervals', 'Action Count', 'Action Distribution Over Iteration Intervals', self.iteration_intervals, self.label_iterations_intervals, self.action_counts)
        plot_actions_from_env(self.config.results_action_counts, self.config.results_action_values, 9)
        print(f"--- Execution took {(time.time() - self.start_time) / 3600} hours ---")
    def get_returns(self):
        return self.returns

    def get_fitness(self):
        return self.fitness

    def get_loss(self):
        return self.loss

    def get_action_counts(self):
        return self.action_counts
