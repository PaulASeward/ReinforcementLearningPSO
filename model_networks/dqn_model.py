import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LSTM
from keras.api._v2.keras.optimizers import Adam

from model_networks.base_model import BaseModel


class DQNModel(BaseModel):
    def __init__(self, config):
        super(DQNModel, self).__init__(config, "dqn")
        self.epsilon = config.epsilon_start

        self.optimizer = Adam(self.config.learning_rate)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

    def nn_model(self):
        model =  tf.keras.Sequential(
            [
                Input((self.config.state_dim,)),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(self.config.action_dim),
            ]
        )

        model.compile(loss=self.compute_loss, optimizer=self.optimizer)

        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):

        self.epsilon *= self.config.epsilon_decay
        self.epsilon = max(self.epsilon, self.config.epsilon_end)

        q_value_array = self.predict(state)
        q_values = q_value_array[0]

        # Could use policies here
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.config.action_dim - 1)

        return np.argmax(q_values)

    def train(self, states, targets):
        history = self.model.fit(states, targets, epochs=1)
        loss = history.history["loss"][0]
        return loss
