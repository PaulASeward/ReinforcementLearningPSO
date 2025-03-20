import tensorflow as tf
import datetime

from coverage.annotate import os
from keras.api._v2.keras.optimizers import Adam, RMSprop, Adagrad, SGD


class BaseModel:
    """
    Base class for deep Q learning
    """
    def __init__(self, config, network_type):
        self.config = config
        self.network_type = network_type
        self.checkpoint_dir = config.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.optimizer = None
        self.debug = not True
        self.sess = None
        self.saver = None

        self.add_optimizer(config.lr_method, config.learning_rate)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

    def add_optimizer(self, lr_method, learning_rate, clip=-1):
        lr_method = lr_method.lower()

        if lr_method == 'adam':  # sgd method
            optimizer = Adam(learning_rate=learning_rate)
        elif lr_method == 'adagrad':
            optimizer = Adagrad(learning_rate=learning_rate)
        elif lr_method == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif lr_method == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate, momentum=0.95, epsilon=0.01)
        else:
            raise NotImplementedError("Unknown method {}".format(lr_method))

        self.optimizer = optimizer

        return optimizer

    def nn_model(self):
        raise NotImplementedError("Method not implemented")

    def save_model(self, step):
        step_dir_path = os.path.join(self.checkpoint_dir, "step_" + str(step))
        if not os.path.exists(step_dir_path):
            os.makedirs(step_dir_path)

        model_step_path = os.path.join(step_dir_path, self.network_type + ".h5")
        self.model.save(model_step_path)

    def load_model(self, step_checkpoint):
        model_step_path = os.path.join(self.checkpoint_dir, str(step_checkpoint), self.network_type + ".h5")
        self.model = tf.keras.models.load_model(model_step_path)

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def get_action_q_values(self, state):
        q_value_array = self.predict(state)
        return q_value_array[0]

