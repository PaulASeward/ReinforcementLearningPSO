import numpy as np
import tensorflow as tf
import os
from datetime import datetime


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(config.log_dir, config.experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.create_file_writer(self.log_dir)


    # def update_target_model_weights(self):
    #     weights = self.model.model.get_weights()
    #     self.target_model.model.set_weights(weights)

