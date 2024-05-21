from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf
from agents.utils.experience_buffer import ExperienceBufferRecurrent as ReplayBuffer
from model_networks.drqn_model import DRQNModel
from logging_utils import ResultsLogger, ComputeDrqnReturn


class DRQNAgent(BaseAgent):
    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)

        self.states = np.zeros([self.config.trace_length, self.config.observation_length])
        self.model = DRQNModel(config)
        self.target_model = DRQNModel(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer()

    def train(self):
        with self.writer.as_default():
            results_logger = ResultsLogger(self.config, self.env, self.raw_env, self.model, ComputeDrqnReturn())
            for ep in range(self.config.train_steps):
                terminal, episode_reward = False, 0
                actions, rewards = [], []

                self.states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4
                current_state = self.env.reset()
                self.update_states(current_state.observation)  # Check states array update

                while not terminal:
                    q_values = self.model.get_action_q_values(np.reshape(self.states, [1, self.config.trace_length, self.config.observation_length]))
                    action = self.policy.select_action(q_values)
                    step_type, reward, discount, next_state = self.env.step(action)

                    reward = reward.numpy()[0]
                    terminal = bool(1 - discount)  # done is 0 (not done) if discount=1.0, and 1 if discount = 0.0

                    actions.append(action)
                    rewards.append(reward)
                    prev_states = self.states

                    self.update_states(next_state)  # Updates the states array removing oldest when adding newest for sliding window

                    self.replay_buffer.add([prev_states, action, reward * self.config.discount_factor, self.states, terminal])

                    episode_reward += reward

                losses = None
                if self.replay_buffer.size() >= self.config.batch_size:
                    losses = self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_model_target_weights()  # target model gets updated AFTER episode, not during like the regular model.

                results_logger.save_log_statements(step=ep+1, actions=actions, rewards=rewards, train_loss=losses)
                print(f"Step #{ep+1} Reward:{episode_reward} Current Epsilon: {self.policy.current_epsilon}")
                # print(f"Actions: {actions}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)

            results_logger.plot_log_statements()