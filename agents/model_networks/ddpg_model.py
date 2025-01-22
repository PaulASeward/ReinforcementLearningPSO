import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Input, Dense, BatchNormalization, Add, Lambda, Concatenate
from agents.model_networks.base_model import BaseModel


class ActorNetworkModel(BaseModel):
    def __init__(self, config):
        # self.action_high = self.config.c_max
        self.action_high = 1
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
        scaling_factor = self.config.upper_bound * np.ones(self.config.action_dimensions)  # TODO: Update action_high into the dynamic scaling max.
        # scaling_factor = self.config.action_dimensions * self.action_high
        output = Lambda(lambda x: x * scaling_factor)(unscaled_output)

        # Shift the output to practical range of the action space
        output = Lambda(lambda x: x + self.config.action_shift)(output)  # TODO: Update the shift to the dynamic shift factors.

        model = Model(inputs=state_input, outputs=output)

        model.compile(loss=self.compute_loss, optimizer=self.optimizer)

        return model

    # def nn_model(self):
    #     # Initialize weights between -3e-3 and 3e-3
    #     init = tf.random_normal_initializer(stddev=0.0005)
    #
    #     model = tf.keras.Sequential(
    #         [
    #             Input((self.config.observation_length,), dtype=tf.float32),
    #             Dense(600, activation=tf.nn.leaky_relu, kernel_initializer=init),
    #             Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=init),
    #             Dense(self.config.action_dimensions, activation="tanh", kernel_initializer=init),
    #         ]
    #     )
    #
    #     # Scale the output by action_high
    #     model.add(Lambda(lambda x: x * self.action_high))
    #
    #     # TODO: IS THIS NEEDED?
    #     model.compile(loss=self.compute_loss, optimizer=self.optimizer)
    #
    #     return model

    @tf.function
    def train(self, X_train, critic_network):
        with tf.GradientTape() as tape:
            y_pred = self.model(X_train, training=True)
            q_pred = critic_network([X_train, y_pred])

            loss_a = -tf.reduce_mean(q_pred)

        actor_grads = tape.gradient(loss_a, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))

        return loss_a


class CriticNetworkModel(BaseModel):
    def __init__(self, config):
        # self.action_high = self.config.c_max
        self.action_high = 1
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

        # TODO: IS THIS NEEDED?
        model.compile(loss=self.compute_loss, optimizer=self.optimizer)
        return model



    # def nn_model(self):
    #     last_init = tf.random_normal_initializer(stddev=0.00005)
    #
    #     # State as input
    #     state_input = Input(shape=(self.config.observation_length,), dtype=tf.float32)
    #     state_out = Dense(600, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(state_input)
    #     state_out = BatchNormalization()(state_out)
    #     state_out = Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(state_out)
    #
    #     # Action as input
    #     action_input = Input(shape=(self.config.action_dimensions,), dtype=tf.float32)
    #     action_out = Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(
    #         Lambda(lambda x: x / self.action_high)(action_input))
    #
    #     # Combine state and action pathways
    #     added = Add()([state_out, action_out])
    #     added = BatchNormalization()(added)
    #
    #     # Further processing layers
    #     outs = Dense(150, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(added)
    #     outs = BatchNormalization()(outs)
    #     outputs = Dense(1, kernel_initializer=last_init)(outs)
    #
    #     # Create the model
    #     model = Model(inputs=[state_input, action_input], outputs=outputs)
    #
    #     model.compile(loss=self.compute_loss, optimizer=self.optimizer)
    #
    #     return model

    def train(self, states, actions, targets):
        history = self.model.fit([states, actions], targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss