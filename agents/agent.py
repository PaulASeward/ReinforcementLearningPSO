import tensorflow as tf
from tf_agents.environments import tf_py_environment
import os
from datetime import datetime
from PSOEnv import PSOEnv
from plot_utils import plot_data_over_iterations, plot_actions_over_iteration_intervals, plot_actions_with_values_over_iteration_intervals
from tf_agents.specs import tensor_spec
import numpy as np


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(config.log_dir, config.experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.raw_env = None
        self.env = None

        self.build_environment()

    # def update_target_model_weights(self):
    #     weights = self.model.model.get_weights()
    #     self.target_model.model.set_weights(weights)

    def build_environment(self):
        minimum = self.config.fDeltas[self.config.func_num - 1]
        environment = PSOEnv(self.config.func_num,
                             minimum=minimum,
                             actions_filename=self.config.env_action_counts,
                             values_filename=self.config.env_action_values,
                             max_episodes=self.config.num_episodes,
                             num_swarm_obs_intervals=self.config.num_swarm_obs_intervals,
                             swarm_obs_interval_length=self.config.swarm_obs_interval_length,
                             swarm_size=self.config.swarm_size,
                             dimension=self.config.dim)
        self.raw_env = environment

        action_tensor_spec = tensor_spec.from_spec(self.raw_env.action_spec())
        self.config.num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        train_environment = tf_py_environment.TFPyEnvironment(environment)
        self.env = train_environment

        return train_environment

    def get_actions(self):
        print(f"num_actions: {self.config.num_actions}")
        action_descriptions = self.raw_env.actions_descriptions

        for index, description in enumerate(action_descriptions):
            action_no = str(index+1)
            print(f"Action #{action_no} Description: {description}")

    def replay_experience(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError

    def build_plots(self):
        plot_data_over_iterations(self.config.average_returns_path, 'Average Return', 'Iteration', self.config.eval_interval)
        plot_data_over_iterations(self.config.fitness_path, 'Average Fitness', 'Iteration', self.config.eval_interval)
        plot_data_over_iterations(self.config.loss_file, 'Average Loss', 'Iteration', self.config.log_interval)
        plot_actions_over_iteration_intervals(self.config.interval_actions_counts_path, 'Iteration Intervals', 'Action Count', 'Action Distribution Over Iteration Intervals', self.config.iteration_intervals, self.config.label_iterations_intervals)
        plot_actions_with_values_over_iteration_intervals(self.config.env_action_counts, self.config.env_action_values, 9)