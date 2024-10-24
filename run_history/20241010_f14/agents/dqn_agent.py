from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf
from agents.utils.experience_buffer import ExperienceBufferStandard as ReplayBuffer
from model_networks.dqn_model import DQNModel
from utils.logging_utils import ResultsLogger


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.states = np.zeros([self.config.trace_length, self.config.observation_length])
        self.model = DQNModel(config)
        self.target_model = DQNModel(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer()

    def train(self):
        with self.writer.as_default():
            results_logger = ResultsLogger(self.config)

            for ep in range(self.config.train_steps):
                terminal, episode_reward = False, 0.0
                actions, rewards = [], []

                self.states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4

                observation = self.env.reset()
                observation = observation.observation

                while not terminal:
                    q_values = self.model.get_action_q_values(observation)
                    action = self.policy.select_action(q_values)
                    actions.append(action)
                    step_type, reward, discount, next_observation = self.env.step(action)

                    reward = reward.numpy()[0]
                    rewards.append(reward)
                    terminal = bool(1 - discount)  # done is 0 (not done) if discount=1.0, and 1 if discount = 0.0

                    self.replay_buffer.add([observation, action, reward * self.config.discount_factor, next_observation, terminal])
                    observation = next_observation
                    episode_reward += reward

                losses = None
                if self.replay_buffer.size() >= self.config.batch_size:
                    losses = self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_model_target_weights()  # target model gets updated AFTER episode, not during like the regular model.

                results_logger.save_log_statements(step=ep+1, actions=actions, rewards=rewards, train_loss=losses)
                print(f"Step #{ep+1} Reward:{episode_reward} Current Epsilon: {self.policy.current_epsilon}")
                # print(f"Actions: {actions}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)
            results_logger.print_execution_time()
