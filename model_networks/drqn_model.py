import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug

from model_networks.base_model import BaseModel


class DRQNModel(BaseModel):
    def __init__(self, n_actions, config):
        super(DRQNModel, self).__init__(config, "drqn")
        self.n_actions = n_actions

        self.lstm_size = config.lstm_size
        self.num_lstm_layers = config.num_lstm_layers
        self.lstm_size = config.lstm_size
        self.min_history = config.min_history
        self.states_to_update = config.states_to_update

