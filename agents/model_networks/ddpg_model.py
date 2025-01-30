import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Input, Dense, BatchNormalization, Lambda, Concatenate
from agents.model_networks.base_model import BaseModel
from tensorflow.keras.optimizers import Adam


class ActorNetworkModel(BaseModel):
    def __init__(self, config):
        super(ActorNetworkModel, self).__init__(config, "actor")

    def nn_model(self):
        # Initialize weights between -3e-3 and 3e-3
        init = tf.random_normal_initializer(stddev=0.0005)

        # State as input
        state_input = Input(shape=self.config.state_shape, dtype=tf.float32)

        # Hidden layers
        x = Dense(self.config.actor_layers[0], name="L0", activation=tf.nn.leaky_relu, kernel_initializer=init)(state_input)
        for index in range(1, len(self.config.actor_layers)):
            x = Dense(self.config.actor_layers[index], name=f"L{index}", activation=tf.nn.leaky_relu, kernel_initializer=init)(x)
            # x = BatchNormalization()(x)

        # Output layer
        unscaled_output = Dense(self.config.action_dimensions, name="Output", activation=tf.nn.tanh)(x)
        scaling_factor = (self.config.upper_bound - self.config.lower_bound) / 2.0
        shift_factor = (self.config.upper_bound + self.config.lower_bound) / 2.0
        output = Lambda(lambda x: x * scaling_factor + shift_factor)(unscaled_output)

        model = Model(inputs=state_input, outputs=output)
        self.optimizer = Adam(learning_rate=self.config.actor_learning_rate)
        # model.compile(loss=self.compute_loss, optimizer=optimizer)

        return model

    @tf.function
    def train(self, states, critic_network):
        with tf.GradientTape() as tape:
            actions = self.model(states, training=True)
            actor_loss = -tf.reduce_mean(critic_network([states, actions]))

        actor_grads = tape.gradient(actor_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))

        return actor_loss


class CriticNetworkModel(BaseModel):
    def __init__(self, config):
        # self.action_high = self.config.c_max
        super(CriticNetworkModel, self).__init__(config, "critic")

    def nn_model(self):
        init = tf.random_normal_initializer(stddev=0.0005)
        action_input_shape = (self.config.action_dimensions,)

        # State as input
        state_input = Input(shape=self.config.state_shape, dtype=tf.float32)
        action_input = Input(shape=action_input_shape, dtype=tf.float32)
        inputs = [state_input, action_input]
        concat = Concatenate(axis=-1)(inputs)

        # Hidden layers
        x = Dense(self.config.critic_layers[0], name="L0", activation=tf.nn.leaky_relu, kernel_initializer=init)(concat)
        for index in range(1, len(self.config.critic_layers)):
            x = Dense(self.config.critic_layers[index], name=f"L{index}", activation=tf.nn.leaky_relu, kernel_initializer=init)(x)
            # x = BatchNormalization()(x)

        output = Dense(1, name="Output", kernel_initializer=init)(x)
        model = Model(inputs=inputs, outputs=output)

        self.optimizer = Adam(learning_rate=self.config.critic_learning_rate)
        # model.compile(loss=self.compute_loss, optimizer=optimizer)
        return model

    @tf.function
    def train(self, states, actions, q_values_targets, ISWeights=1.0):
        with tf.GradientTape() as tape:
            q_values = self.model([states, actions])
            td_error = q_values - q_values_targets
            critic_loss = tf.reduce_mean(ISWeights * tf.math.square(td_error))

        critic_grad = tape.gradient(critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grad, self.model.trainable_variables))
        return critic_loss, td_error
