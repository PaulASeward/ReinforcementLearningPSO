import numpy as np
import os
import tensorflow as tf
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug

from model_networks.base_model import BaseModel


class DRQNModel(BaseModel):
    def __init__(self, config):
        super(DRQNModel, self).__init__(config, "drqn")
        self.epsilon = config.epsilon_start

        self.optimizer = Adam(self.config.learning_rate)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

        self.num_lstm_layers = config.num_lstm_layers
        self.lstm_size = config.lstm_size
        self.min_history = config.min_history
        self.states_to_update = config.states_to_update

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((self.config.trace_len, self.config.state_dim)),
                LSTM(32, activation="tanh"),
                Dense(16, activation="relu"),
                Dense(self.config.action_dim),
            ]
        )

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.config.trace_length, self.config.state_dim])

        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.epsilon, self.config.epsilon_end)

        q_value = self.predict(state)[0]

        # Could use policies here
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.config.action_dim - 1)

        return np.argmax(q_value)

    def train(self, states, targets):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)

            assert targets.shape == logits.shape

            loss = self.compute_loss(targets, logits)
            grads = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))