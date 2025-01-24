from agents.agent import BaseAgent
import numpy as np
from environment.gym_env_continuous import ContinuousPsoGymEnv
from environment.gym_multiswarm_env_continuous import ContinuousMultiSwarmPsoGymEnv
from tf_agents.environments import tf_py_environment

import gymnasium as gym
from agents.utils.experience_buffer import ExperienceBufferStandard as ReplayBuffer
from agents.model_networks.ddpg_model import ActorNetworkModel, CriticNetworkModel
from utils.logging_utils import ContinuousActionsResultsLogger as ResultsLogger
from agents.utils.policy import OrnsteinUhlenbeckActionNoisePolicy, OrnsteinUhlenbeckActionNoisePolicyWithDecayScaling


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        super(DDPGAgent, self).__init__(config)
        # TODO: Adjust states from 50 particles with 3 dimensions = 150 flattened array to 50 particles with 3 dimensions each.
        self.results_logger = ResultsLogger(config)

        self.actor_network = ActorNetworkModel(config)
        self.actor_network_target = ActorNetworkModel(config)

        self.critic_network = CriticNetworkModel(config)
        self.critic_network_target = CriticNetworkModel(config)
        # self.policy = OrnsteinUhlenbeckActionNoisePolicy(config)
        self.policy = OrnsteinUhlenbeckActionNoisePolicyWithDecayScaling(config)

        self.update_model_target_weights(tau=1.0)
        self.replay_buffer = ReplayBuffer()

    def build_environment(self):
        if self.config.swarm_algorithm == "PMSO":
            # low_limit_subswarm_action_space = [-(self.config.w - self.config.w_min), -(self.config.c1 - self.config.c_min),
            #                                    -(self.config.c2 - self.config.c_min)]
            # high_limit_subswarm_action_space = [self.config.w_max - self.config.w, self.config.c_max - self.config.c1,
            #                                     self.config.c_max - self.config.c2]
            low_limit_subswarm_action_space = [self.config.w_min, self.config.c_min, self.config.c_min]
            high_limit_subswarm_action_space = [self.config.w_max, self.config.c_max, self.config.c_max]

            self.config.lower_bound = np.array(
                [low_limit_subswarm_action_space for _ in range(self.config.num_sub_swarms)], dtype=np.float32).flatten()
            self.config.upper_bound = np.array(
                [high_limit_subswarm_action_space for _ in range(self.config.num_sub_swarms)], dtype=np.float32).flatten()

            if self.config.use_mock_data:
                self.raw_env = gym.make("MockContinuousPmsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
            else:
                self.raw_env = gym.make("ContinuousMultiSwarmPsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
        else:
            # self.config.lower_bound = np.array(
            #     [-(self.config.w - self.config.w_min), -(self.config.c1 - self.config.c_min), -(self.config.c2 - self.config.c_min)], dtype=np.float32)
            # self.config.upper_bound = np.array(
            #     [self.config.w_max - self.config.w, self.config.c_max - self.config.c1, self.config.c_max - self.config.c2], dtype=np.float32)
            self.config.lower_bound = np.array([self.config.w_min, self.config.c_min, self.config.c_min], dtype=np.float32)
            self.config.upper_bound = np.array([self.config.w_max, self.config.c_max, self.config.c_max], dtype=np.float32)

            if self.config.use_mock_data:
                self.raw_env = gym.make("MockContinuousPsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
            else:
                self.raw_env = gym.make("ContinuousPsoGymEnv-v0", config=self.config)
                self.env = self.raw_env

        self.config.state_shape = self.env.observation_space.shape


    def get_q_values(self, state):
        return self.actor_network.get_action_q_values(state)

    def update_model_target_weights(self, tau=None):
        if tau is None:
            tau = self.config.tau

        if not self.config.use_mock_data:
            # Get the weights of the actor and critic networks
            theta_a, theta_c, theta_a_targ, theta_c_targ = self.actor_network.model.get_weights(), self.critic_network.model.get_weights(), self.actor_network_target.model.get_weights(), self.critic_network_target.model.get_weights()

            # mixing factor tau : we gradually shift the weights...
            theta_a_targ = [theta_a[i] * tau + theta_a_targ[i] * (1 - tau) for i in range(len(theta_a))]
            theta_c_targ = [theta_c[i] * tau + theta_c_targ[i] * (1 - tau) for i in range(len(theta_c))]

            self.actor_network_target.model.set_weights(theta_a_targ)
            self.critic_network_target.model.set_weights(theta_c_targ)

    def replay_experience(self, experience_length=5):
        if self.replay_buffer.size() < self.config.batch_size:
            return None  # Not enough experience to replay yet.

        losses = []
        if not self.config.use_mock_data:
            for _ in range(experience_length):
                states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)

                # ---------------------------- update critic ---------------------------- #
                next_actions = self.actor_network_target.predict(next_states)
                next_q_values = self.critic_network_target.predict([next_states, next_actions])

                # Use Bellman Equation. (recursive definition of q-values)
                q_values_targets = rewards + (1 - done) * self.config.gamma * next_q_values
                q_values_targets = q_values_targets.astype(np.float32)

                critic_loss = self.critic_network.train(states, actions, q_values_targets)

                # ---------------------------- update actor ---------------------------- #
                action_loss = self.actor_network.train(states, self.critic_network.model)

                losses.append(action_loss + critic_loss)

        return losses


    # def replay_experience(self, experience_length=10):
    #     if self.replay_buffer.size() < self.config.batch_size:
    #         return None  # Not enough experience to replay yet.
    #
    #     losses = []
    #     if not self.config.use_mock_data:
    #         for _ in range(experience_length):  # Why size 10?
    #             states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)
    #
    #             # ---------------------------- update critic ---------------------------- #
    #             next_actions = self.actor_network_target.model(next_states)
    #             next_q_values = self.critic_network_target.model([next_states, next_actions])
    #
    #             # Use Bellman Equation. (recursive definition of q-values)
    #             q_values_target = rewards + (1 - done) * self.config.gamma * next_q_values
    #
    #             self.critic_network.model.fit([states, actions], q_values_target, batch_size=self.config.batch_size, epochs=1, verbose=0, shuffle=False)
    #
    #             # ---------------------------- update actor ---------------------------- #
    #             action_loss = self.actor_network.train(states, self.critic_network.model)
    #
    #             losses.append(action_loss)
    #
    #     return losses
