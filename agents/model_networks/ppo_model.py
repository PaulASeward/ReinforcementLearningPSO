import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Input, Dense, BatchNormalization, Lambda, Concatenate, Attention, LayerNormalization
from agents.model_networks.base_model import BaseModel
from tensorflow.keras.optimizers import Adam


class ActorNetworkModel(BaseModel):
    def __init__(self, config):
        super(ActorNetworkModel, self).__init__(config, "actor")

    def nn_model(self):
        initial_input = Input(shape=self.config.state_shape, dtype=tf.float32)

        if self.config.use_attention_layer:
            attention_output = Attention(use_scale=True)([initial_input, initial_input])
            first_input = LayerNormalization()(attention_output)
        else:
            first_input = initial_input

        x = Dense(self.config.actor_layers[0], name="L0", activation='tanh')(first_input)
        for index in range(1, len(self.config.actor_layers)):
            x = Dense(self.config.actor_layers[index], name=f"L{index}", activation='tanh')(x)
            # x = BatchNormalization()(x)

        # For continuous actions, output a mean plus log_std
        mean = Dense(self.config.action_dimensions, activation=None)(x)
        log_std = tf.Variable(initial_value=-0.5 * tf.ones(self.config.action_dimensions, dtype=tf.float32),
                              trainable=True, name="log_std")

        model = Model(inputs=initial_input, outputs=[mean, log_std])
        self.optimizer = Adam(learning_rate=self.config.actor_learning_rate)

        return model

    def sample_action(self, obs):
        """
        Samples an action from the current policy (Gaussian).
        Returns the action and log-prob of that action under this policy.
        """
        mean, log_std = self.model(obs, training=False)
        std = tf.exp(log_std)
        pi_distribution = tf.random.normal(shape=tf.shape(mean)) * std + mean
        logp = self.gaussian_likelihood(pi_distribution, mean, log_std)

        return pi_distribution, logp

    @staticmethod
    def gaussian_likelihood(x, mu, log_std):
        """
        Compute log probability of x under Diagonal Gaussian with mean mu and log-std log_std.
        """
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2.0 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @tf.function
    def train(self, obs, act, logp_old, adv):
        """
        The PPO clipping objective for the policy.
        """
        with tf.GradientTape() as tape:
            mean, log_std = self.model(obs, training=True)
            logp = self.gaussian_likelihood(act, mean, log_std)
            ratio = tf.exp(logp - logp_old)

            # PPO objective
            clip_adv = tf.where(
                adv >= 0,
                (1 + self.config.clip_ratio) * adv,
                (1 - self.config.clip_ratio) * adv
            )
            obj = tf.minimum(ratio * adv, clip_adv)
            loss_pi = -tf.reduce_mean(obj)

            # Approximate KL for early stopping
            approx_kl = tf.reduce_mean(logp_old - logp)

        grads = tape.gradient(loss_pi, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_pi, approx_kl


class CriticNetworkModel(BaseModel):
    def __init__(self, config):
        super(CriticNetworkModel, self).__init__(config, "critic")

    def nn_model(self):
        initial_input = Input(shape=self.config.state_shape, dtype=tf.float32)

        if self.config.use_attention_layer:
            attention_output = Attention(use_scale=True)([initial_input, initial_input])
            first_input = LayerNormalization()(attention_output)
        else:
            first_input = initial_input

        x = Dense(self.config.critic_layers[0], name="L0", activation='tanh')(first_input)
        for index in range(1, len(self.config.critic_layers)-1):
            x = Dense(self.config.critic_layers[index], name=f"L{index}", activation='tanh')(x)
            # x = BatchNormalization()(x)
        output = Dense(units=self.config.critic_layers[-1], name='Output', activation=None)(x)

        model = Model(inputs=initial_input, outputs=output)
        self.optimizer = Adam(learning_rate=self.config.critic_learning_rate)

        return model

    @tf.function
    def train(self, obs, ret):
        """
        Simple MSE loss for value function: L = mean( (V(s) - ret)^2 ).
        """
        with tf.GradientTape() as tape:
            value_pred = self.model(obs, training=True)
            loss_v = tf.reduce_mean((ret - value_pred) ** 2)

        grads = tape.gradient(loss_v, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_v