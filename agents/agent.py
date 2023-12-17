import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
import os
from datetime import datetime
from PSOEnv import PSOEnv

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(config.log_dir, config.experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.create_file_writer(self.log_dir)

        self.raw_env = None
        self.env = None

    # def update_target_model_weights(self):
    #     weights = self.model.model.get_weights()
    #     self.target_model.model.set_weights(weights)

    def build_environment(self):
        minimum = self.config.fDeltas[self.config.func_num -1]
        environment = PSOEnv(self.config.func_num, dimension=self.config.dimension, minimum=minimum)
        self.raw_env = environment

        train_environment = tf_py_environment.TFPyEnvironment(environment)
        self.env = train_environment

        return train_environment

    def replay_experience(self):
        raise NotImplementedError

    def train(self, max_episodes=1000):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError