import gym
import tensorflow as tf
from tf_agents.environments import tf_py_environment, wrappers
import os
import numpy as np
from datetime import datetime
from environment.mock.mock_env import MockEnv
import environment.gym_env_continuous
import environment.mock.mock_gym_env_continuous
import environment.mock.mock_gym_env_discrete
from environment.gym_env_discrete import DiscretePsoGymEnv

from environment.env import PSOEnv
from utils.plot_utils import plot_data_over_iterations, plot_actions_over_iteration_intervals, plot_actions_with_values_over_iteration_intervals, plot_continuous_actions_over_iteration_intervals
from agents.utils.policy import ExponentialDecayGreedyEpsilonPolicy


class BaseAgent:
    def __init__(self, config):
        self.replay_buffer = None
        self.states = None
        self.target_model = None
        self.model = None
        self.policy = None
        self.config = config
        self.log_dir = os.path.join(config.log_dir, config.experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.raw_env = None
        self.env = None

        self.build_environment()
        if config.policy == "ExponentialDecayGreedyEpsilon":
            self.set_policy(ExponentialDecayGreedyEpsilonPolicy(epsilon_start=config.epsilon_start, epsilon_end=config.epsilon_end, num_steps=config.train_steps, num_actions=config.num_actions))

    def set_policy(self, policy):
        self.policy = policy

    def update_model_target_weights(self):
        if not self.config.use_mock_data:
            weights = self.model.model.get_weights()
            self.target_model.model.set_weights(weights)

    # def update_states(self, next_state):
    #     self.states = np.roll(self.states, -1, axis=0)
    #     self.states[-1] = next_state

    def replay_experience(self, experience_length=10):
        losses = []
        if not self.config.use_mock_data:
            for _ in range(experience_length):  # Why size 10?
                states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)
                targets = self.target_model.predict(states)

                next_q_values = self.target_model.predict(next_states).max(axis=1)
                targets[range(self.config.batch_size), actions] = (rewards + (1 - done) * next_q_values * self.config.gamma)

                loss = self.model.train(states, targets)
                losses.append(loss)

        return losses

    def build_environment(self):
        if self.config.network_type == "DDPG" and not self.config.discrete_action_space:
            if self.config.use_mock_data:
                self.raw_env = gym.make("MockContinuousPsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
            else:
                self.raw_env = gym.make("ContinuousMultiSwarmPsoGymEnv-v0", config=self.config)
                # self.raw_env = gym.make("ContinuousPsoGymEnv-v0", config=self.config)  # Continuous environment
                self.env = self.raw_env
        elif self.config.network_type == "DQN":
            if self.config.use_mock_data:
                self.raw_env = gym.make("MockDiscretePsoGymEnv-v0", config=self.config)  # Continuous environment
                self.env = self.raw_env
            else:
                self.raw_env = gym.make("DiscretePsoGymEnv-v0", config=self.config)  # Continuous environment
                self.env = self.raw_env
        else:
            if self.config.use_mock_data:
                self.raw_env = MockEnv(self.config)  # Mock environment
                self.env = tf_py_environment.TFPyEnvironment(self.raw_env)  # Training environment
            else:
                self.raw_env = PSOEnv(self.config)  # Raw environment
                self.env = tf_py_environment.TFPyEnvironment(self.raw_env)  # Training environment

        return self.env

    def get_actions(self):
        print(f"num_actions: {self.config.num_actions}")

        for index, description in enumerate(self.raw_env.actions_descriptions):
            if index +1 <= self.config.num_actions:
                action_no = str(index+1)
                print(f"Action #{action_no} Description: {description}")

    def build_plots(self):
        plot_data_over_iterations(self.config.average_returns_path, 'Average Return', 'Iteration', self.config.eval_interval)
        plot_data_over_iterations(self.config.fitness_path, 'Average Fitness', 'Iteration', self.config.eval_interval)
        plot_data_over_iterations(self.config.loss_file, 'Average Loss', 'Iteration', self.config.log_interval)
        if self.config.discrete_action_space:
            plot_actions_over_iteration_intervals(self.config.interval_actions_counts_path, self.config.fitness_path, 'Iteration Intervals', 'Action Count', 'Action Distribution Over Iteration Intervals', self.config.iteration_intervals, self.config.label_iterations_intervals, self.raw_env.actions_descriptions)
            plot_actions_with_values_over_iteration_intervals(self.config.action_counts_path, self.config.action_values_path, standard_pso_values_path=self.config.standard_pso_path, function_min_value=self.config.fDeltas[self.config.func_num - 1], num_actions=self.config.num_actions, action_names=self.raw_env.actions_descriptions)
        else:
            plot_continuous_actions_over_iteration_intervals(self.config.action_counts_path, self.config.action_values_path, standard_pso_values_path=self.config.standard_pso_path, function_min_value=self.config.fDeltas[self.config.func_num - 1], num_actions=self.config.num_actions, action_names=self.raw_env.actions_descriptions, action_offset=self.raw_env.actions_offset, num_intervals=18)