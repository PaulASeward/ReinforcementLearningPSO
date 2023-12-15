import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug

from model_networks.base_model import BaseModel


class DQNModel(BaseModel):
    def __init__(self, n_actions, config):
        super(DQNModel, self).__init__(config, "dqn")
        self.n_actions = n_actions
        self.history_len = config.history_len


