import numpy as np
import tensorflow as tf
from keras.api._v2.keras.optimizers import Adam
from keras.layers import Input, Dense, LSTM
from model_networks.base_model import BaseModel


class DRQNModel(BaseModel):
    def __init__(self, config):
        super(DRQNModel, self).__init__(config, "drqn")
        self.epsilon = config.epsilon_start
        self.add_optimizer(config.lr_method, config.learning_rate)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

        self.num_lstm_layers = config.num_lstm_layers
        self.lstm_size = config.lstm_size
        self.min_history = config.min_history
        self.states_to_update = config.states_to_update

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((self.config.trace_length, self.config.observation_length)),
                LSTM(32, activation="tanh"),
                Dense(16, activation="relu"),
                Dense(self.config.num_actions),
            ]
        )

    # def predict(self, state):
    #     return self.model.predict(state)
    #
    # def get_action_q_values(self, state):
    #     q_value_array = self.predict(state)
    #     return q_value_array[0]
    #
    #     # Could use policies here
    #     self.epsilon *= self.config.epsilon_decay
    #     self.epsilon = max(self.epsilon, self.config.epsilon_end)
    #
    #     if np.random.random() < self.epsilon:
    #         return np.random.randint(0, self.config.num_actions - 1)
    #
    #     return np.argmax(q_value)

    def train(self, states, targets):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)

            assert targets.shape == logits.shape

            loss = self.compute_loss(targets, logits)
            grads = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            loss = loss.numpy()

            return loss