import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LSTM
from keras.api._v2.keras.optimizers import Adam

from model_networks.base_model import BaseModel


class DQNModel(BaseModel):
    def __init__(self, config):
        super(DQNModel, self).__init__(config, "dqn")
        self.epsilon = config.epsilon_start

        self.add_optimizer(config.lr_method, config.learning_rate)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

    def nn_model(self):
        model = tf.keras.Sequential(
            [
                Input((self.config.observation_length,)),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(self.config.num_actions),
            ]
        )

        model.compile(loss=self.compute_loss, optimizer=self.optimizer)

        return model

    # def predict(self, state):
    #     return self.model.predict(state)
    #
    # def get_action_q_values(self, state):
    #     q_value_array = self.predict(state)
    #     q_values = q_value_array[0]
    #
    #     # Could use policies here
    #     self.epsilon *= self.config.epsilon_decay
    #     self.epsilon = max(self.epsilon, self.config.epsilon_end)
    #
    #     if np.random.random() < self.epsilon:
    #         return np.random.randint(0, self.config.num_actions - 1)
    #
    #     return np.argmax(q_values)

    def train(self, states, targets):
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss
