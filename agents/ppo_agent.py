from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf

from environment.gym_env_continuous import ContinuousPsoGymEnv
from tf_agents.environments import tf_py_environment

import gymnasium as gym
from agents.utils.experience_buffer import PPOBuffer
from agents.model_networks.ppo_model import ActorNetworkModel, CriticNetworkModel
from utils.logging_utils import ContinuousActionsResultsLogger as ResultsLogger
from agents.utils.policy import PPOPolicy


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super(PPOAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)

        self.actor_network = ActorNetworkModel(config)
        self.critic_network = CriticNetworkModel(config)
        self.policy = PPOPolicy(config, actor_network=self.actor_network)

        self.current_kl = None
        self.current_value = None
        self.trajectory_buffer = PPOBuffer(config)

    def replay_experience(self):
        """
        Perform the PPO update: fetch the entire on-policy batch from PPOBuffer,
        then run gradient descent for a certain number of epochs on policy and value networks.
        """
        obs, act, adv, ret, logp_old  = self.trajectory_buffer.get()
        pi_losses, v_losses, kls = [], [], []

        # Update the policy for a certain number of iterations
        for _ in range(self.config.train_policy_iterations):
            loss_pi, kl = self.actor_network.train(obs, act, logp_old, adv)
            pi_losses.append(loss_pi)
            kls.append(kl)
            if kl > 1.5 * self.config.target_kl:
                # Early stopping if KL grows too big
                break

        # Update the value function for a certain number of iterations
        for _ in range(self.config.train_value_iterations):
            loss_v = self.critic_network.train(obs, ret)
            v_losses.append(loss_v)

        kls = np.mean(kls if len(kls) > 0 else [0.0])
        self.current_kl = kls

        pi_loss, value_loss = np.mean(pi_losses), np.mean(v_losses)
        total_loss = pi_loss + value_loss
        return total_loss, pi_loss, value_loss

    def update_model_target_weights(self):
        self.trajectory_buffer.finish_path(0)

        if self.current_kl > 1.5 * self.config.target_kl:
            print(f"Early stopping due to reaching max KL.")
            print(f"Current KL: {self.current_kl}")
            print(f"Target KL: {self.config.target_kl}")
            print(f"Current KL > 1.5 * Target KL: {self.current_kl > 1.5 * self.config.target_kl}")
            return True
        return False

    def initialize_current_state(self):
        # self.policy.reset()
        observation, swarm_info = self.env.reset()
        return tf.convert_to_tensor(observation.reshape(1, -1), dtype=tf.float32)

    def update_memory_and_state(self, current_state, action, reward, next_observation, terminal):
        # TODO: How do I better access value and logp?
        value = self.current_value
        logp = self.policy.logp

        next_state = tf.convert_to_tensor(next_observation.reshape(1, -1), dtype=tf.float32)
        self.trajectory_buffer.store(current_state, action, reward, value, logp)
        return next_state

    def get_q_values(self, state):
        value = self.critic_network.predict(state)
        self.current_value = value[0]
        return state

