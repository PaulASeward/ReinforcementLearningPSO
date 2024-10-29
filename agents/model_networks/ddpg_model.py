import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Add, Lambda
from agents.model_networks.base_model import BaseModel


class ActorNetworkModel(BaseModel):
    def __init__(self, config, action_high=1):
        super(ActorNetworkModel, self).__init__(config, "actor")
        self.action_high = action_high

    def nn_model(self):
        # Initialize weights between -3e-3 and 3e-3
        last_init = tf.random_normal_initializer(stddev=0.0005)

        model = tf.keras.Sequential(
            [
                Input((self.config.num_states,), dtype=tf.float32),
                Dense(600, activation=tf.nn.leaky_relu, kernel_initializer=last_init),
                Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=last_init),
                Dense(self.config.num_actions, activation="tanh", kernel_initializer=last_init),
            ]
        )

        # Scale the output by action_high
        model.add(tf.keras.layers.Lambda(lambda x: x * self.action_high))

        return model

    def train(self, states, targets):
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss


class CriticNetworkModel(BaseModel):
    def __init__(self, config, action_high=1):
        super(CriticNetworkModel, self).__init__(config, "critic")
        self.action_high = action_high

    def nn_model(self):
        last_init = tf.random_normal_initializer(stddev=0.00005)

        # State as input
        state_input = Input(shape=(self.config.num_states,), dtype=tf.float32)
        state_out = Dense(600, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(state_input)
        state_out = BatchNormalization()(state_out)
        state_out = Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(state_out)

        # Action as input
        action_input = Input(shape=(self.config.num_actions,), dtype=tf.float32)
        action_out = Dense(300, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(
            Lambda(lambda x: x / self.action_high)(action_input))

        # Combine state and action pathways
        added = Add()([state_out, action_out])
        added = BatchNormalization()(added)

        # Further processing layers
        outs = Dense(150, activation=tf.nn.leaky_relu, kernel_initializer=last_init)(added)
        outs = BatchNormalization()(outs)
        outputs = Dense(1, kernel_initializer=last_init)(outs)

        # Create the model
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
        return model

    def train(self, states, actions, targets):
        history = self.model.fit([states, actions], targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss


class DDPGModel(BaseModel):
    def __init__(self, config):
        super(DDPGModel, self).__init__(config, "ddpg")

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

    def train(self, states, targets):
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history["loss"][0]
        return loss
