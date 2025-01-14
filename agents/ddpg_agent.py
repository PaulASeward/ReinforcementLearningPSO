from agents.agent import BaseAgent
import numpy as np
from environment.gym_env_continuous import ContinuousPsoGymEnv
from environment.gym_multiswarm_env_continuous import ContinuousMultiSwarmPsoGymEnv
from tf_agents.environments import tf_py_environment

import gymnasium as gym
from agents.utils.experience_buffer import ExperienceBufferStandard as ReplayBuffer
from agents.model_networks.ddpg_model import ActorNetworkModel, CriticNetworkModel
from utils.logging_utils import ContinuousActionsResultsLogger as ResultsLogger
from agents.utils.policy import OrnsteinUhlenbeckActionNoisePolicy


class DDPGAgent(BaseAgent):
    def __init__(self, config):
        super(DDPGAgent, self).__init__(config)
        # TODO: Adjust states from 50 particles with 3 dimensions = 150 flattened array to 50 particles with 3 dimensions each.
        self.results_logger = ResultsLogger(config)

        self.states = np.zeros([self.config.trace_length, self.config.observation_length])

        self.actor_network = ActorNetworkModel(config)
        self.actor_network_target = ActorNetworkModel(config)

        self.critic_network = CriticNetworkModel(config)
        self.critic_network_target = CriticNetworkModel(config)
        self.policy = OrnsteinUhlenbeckActionNoisePolicy(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer()

    def build_environment(self):
        if self.config.swarm_algorithm == "PMSO":
            low_limit_subswarm_action_space = [-(self.config.w - self.config.w_min), -(self.config.c1 - self.config.c_min),
                                               -(self.config.c2 - self.config.c_min)]
            high_limit_subswarm_action_space = [self.config.w_max - self.config.w, self.config.c_max - self.config.c1,
                                                self.config.c_max - self.config.c2]
            self.config.lower_bound = np.array(
                [low_limit_subswarm_action_space for _ in range(self.config.num_sub_swarms)]).flatten()
            self.config.upper_bound = np.array(
                [high_limit_subswarm_action_space for _ in range(self.config.num_sub_swarms)]).flatten()

            if self.config.use_mock_data:
                self.raw_env = gym.make("MockContinuousPmsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
            else:
                self.raw_env = gym.make("ContinuousMultiSwarmPsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
        else:
            self.config.lower_bound = np.array(
                [-(self.config.w - self.config.w_min), -(self.config.c1 - self.config.c_min), -(self.config.c2 - self.config.c_min)])
            self.config.upper_bound = np.array(
                [self.config.w_max - self.config.w, self.config.c_max - self.config.c1, self.config.c_max - self.config.c2])

            if self.config.use_mock_data:
                self.raw_env = gym.make("MockContinuousPsoGymEnv-v0", config=self.config)
                self.env = self.raw_env
            else:
                self.raw_env = gym.make("ContinuousPsoGymEnv-v0", config=self.config)
                self.env = self.raw_env

    def get_actions(self):
        print(f"num_actions: {self.config.action_dimensions}")

        for index, description in enumerate(self.config.actions_descriptions):
            if index + 1 <= self.config.action_dimensions:
                action_no = str(index + 1)
                print(f"Action #{action_no} Description: {description}")

    def get_q_values(self, state):
        return self.actor_network.get_action_q_values(state)

    def update_model_target_weights(self):
        if not self.config.use_mock_data:
            # Get the weights of the actor and critic networks
            theta_a, theta_c, theta_a_targ, theta_c_targ = self.actor_network.model.get_weights(), self.critic_network.model.get_weights(), self.actor_network_target.model.get_weights(), self.critic_network_target.model.get_weights()

            # mixing factor tau : we gradually shift the weights...
            theta_a_targ = [theta_a[i] * self.config.tau + theta_a_targ[i] * (1 - self.config.tau) for i in range(len(theta_a))]
            theta_c_targ = [theta_c[i] * self.config.tau + theta_c_targ[i] * (1 - self.config.tau) for i in range(len(theta_c))]

            self.actor_network_target.model.set_weights(theta_a_targ)
            self.critic_network_target.model.set_weights(theta_c_targ)

    def replay_experience(self, experience_length=10):
        if self.replay_buffer.size() < self.config.batch_size:
            return None  # Not enough experience to replay yet.

        losses = []
        if not self.config.use_mock_data:
            for _ in range(experience_length):  # Why size 10?
                states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)

                # ---------------------------- update critic ---------------------------- #
                next_actions = self.actor_network_target.model(next_states)
                next_q_values = self.critic_network_target.model([next_states, next_actions])

                # Use Bellman Equation. (recursive definition of q-values)
                q_values_target = rewards + (1 - done) * self.config.gamma * next_q_values

                self.critic_network.model.fit([states, actions], q_values_target, batch_size=self.config.batch_size, epochs=1, verbose=0, shuffle=False)

                # ---------------------------- update actor ---------------------------- #
                action_loss = self.actor_network.train(states, self.critic_network.model)

                losses.append(action_loss)

        return losses
