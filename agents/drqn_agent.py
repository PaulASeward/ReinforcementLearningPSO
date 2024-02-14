from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf
from experience_buffer import ExperienceBufferRecurrentTutorial as ReplayBuffer
from model_networks.drqn_model import DRQNModel
from logging_utils import ResultsLogger, ComputeDrqnReturn


class DRQNAgent(BaseAgent):
    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)

        self.states = np.zeros([self.config.trace_length, self.config.observation_length])
        self.model = DRQNModel(config)
        self.target_model = DRQNModel(config)

        self.update_target()
        self.replay_buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def replay_experience(self, experience_length=10):
        losses = []
        for _ in range(experience_length):  # Why size 10?
            states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)
            targets = self.target_model.predict(states)  # This is likely unnecessary as it ges rewritten

            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(self.config.batch_size), actions] = (rewards + (1 - done) * next_q_values * self.config.gamma)

            loss = self.model.train(states, targets)
            loss = loss.numpy()
            losses.append(loss)

        return losses

    def train(self):
        with self.writer.as_default():
            results_logger = ResultsLogger(self.config, self.env, self.model, ComputeDrqnReturn())
            for ep in range(self.config.train_steps):
                done, episode_reward, actions = False, 0, []
                actions, rewards = [], []

                self.states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4
                current_state = self.env.reset()
                self.update_states(current_state.observation)  # Check states array update

                while not done:
                    action = self.model.get_action(self.states)
                    step_type, reward, discount, next_state = self.env.step(action)

                    reward = reward.numpy()[0]
                    done = bool(1 - discount)  # done is 0 (not done) if discount=1.0, and 1 if discount = 0.0

                    actions.append(action)
                    rewards.append(reward)
                    prev_states = self.states

                    self.update_states(next_state)  # Updates the states array removing oldest when adding newest for sliding window

                    self.replay_buffer.add([prev_states, action, reward * self.config.discount_factor, self.states, done])

                    episode_reward += reward

                losses = None
                if self.replay_buffer.size() >= self.config.batch_size:
                    losses = self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_target()  # target model gets updated AFTER episode, not during like the regular model.

                results_logger.save_log_statements(step=ep+1, actions=actions, train_loss=losses)
                print(f"Step #{ep+1} Reward:{episode_reward}")
                # print(f"Actions: {actions}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)

            results_logger.plot_log_statements()