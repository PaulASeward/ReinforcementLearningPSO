from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf
from agents.utils.experience_buffer import ExperienceBufferStandard as ReplayBuffer
from agents.model_networks.ddpg_model import ActorNetworkModel, CriticNetworkModel
from utils.logging_utils import ResultsLogger
from agents.utils.policy import OrnsteinUhlenbeckActionNoisePolicy
# from agents.utils.noise import OrnsteinUhlenbeckActionNoise

class DDPGAgent(BaseAgent):
    def __init__(self, config):
        super(DDPGAgent, self).__init__(config)
        self.states = np.zeros([self.config.trace_length, self.config.observation_length])

        self.actor_network = ActorNetworkModel(config)
        self.actor_network_target = ActorNetworkModel(config)

        self.critic_network = CriticNetworkModel(config)
        self.critic_network_target = CriticNetworkModel(config)
        self.set_policy(OrnsteinUhlenbeckActionNoisePolicy(config))

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer()

    def update_model_target_weights(self):
        if not self.config.use_mock_data:
            theta_a, theta_c = self.actor_network.model.get_weights(), self.critic_network.model.get_weights()
            theta_a_targ, theta_c_targ = self.actor_network_target.model.get_weights(), self.critic_network_target.model.get_weights()

            # mixing factor tau : we gradually shift the weights...
            theta_a_targ = [theta_a[i] * self.config.tau + theta_a_targ[i] * (1 - self.config.tau) for i in range(len(theta_a))]
            theta_c_targ = [theta_c[i] * self.config.tau + theta_c_targ[i] * (1 - self.config.tau) for i in range(len(theta_c))]

            self.actor_network_target.model.set_weights(theta_a_targ)
            self.critic_network_target.model.set_weights(theta_c_targ)

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def replay_experience(self, experience_length=10):
        losses = []
        if not self.config.use_mock_data:
            for _ in range(experience_length):  # Why size 10?
                states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)

                # ---------------------------- update critic ---------------------------- #
                next_actions = self.actor_network_target.model(next_states)
                next_q_values = self.critic_network_target.model([next_states, next_actions])

                # Use Bellman Equation! (recursive definition of q-values)
                q_values_target = rewards + (1 - done) * self.config.gamma * next_q_values

                self.critic_network.model.fit([states, actions], q_values_target, batch_size=self.config.batch_size, epochs=1, verbose=0, shuffle=False)

                # ---------------------------- update actor ---------------------------- #
                action_loss = self.actor_network.train(states, self.critic_network.model)

                losses.append(action_loss)

        return losses

    def train(self):
        with self.writer.as_default():
            results_logger = ResultsLogger(self.config)

            for ep in range(self.config.train_steps):
                terminal, episode_reward = False, 0.0
                actions, rewards = [], []

                self.states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4

                self.policy.reset()  # Reset the noise process
                observation = self.env.reset()
                observation = observation[0]
                state = np.reshape(observation, (1, self.config.observation_length))

                while not terminal:
                    q_values = self.actor_network.get_action_q_values(state)
                    action = self.policy.select_action(q_values)  # Doesn't actually select one discrete action, but adds noise to the continuous action space.
                    next_observation, reward, terminal, info = self.env.step(action)

                    next_state = np.reshape(next_observation, (1, self.config.observation_length))
                    self.replay_buffer.add([state, action, reward * self.config.discount_factor, next_state, terminal])

                    state = next_state
                    episode_reward += reward
                    actions.append(action)
                    rewards.append(reward)

                losses = None
                if self.replay_buffer.size() >= self.config.batch_size:
                    losses = self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_model_target_weights()  # target model gets updated AFTER episode, not during like the regular model.

                results_logger.save_log_statements(step=ep+1, actions=actions, rewards=rewards, train_loss=losses)
                print(f"Step #{ep+1} Reward:{episode_reward} Current Epsilon: {self.policy.current_epsilon}")
                # print(f"Actions: {actions}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)
            results_logger.print_execution_time()
