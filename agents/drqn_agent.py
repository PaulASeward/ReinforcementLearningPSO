from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf
from experience_buffer import RecurrentExperienceBuffer as ReplayBuffer

# from tqdm import tqdm
from model_networks.drqn_model import DRQNModel


class DRQNAgent(BaseAgent):
    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)

        self.states = np.zeros([self.config.trace_length, self.config.state_dim])
        self.model = DRQNModel(config)
        self.target_model = DRQNModel(config)

        self.update_target()
        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def replay_experience(self):
        for _ in range(10):  # Why size 10?
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)  # This is likely unnecessary as it ges rewritten

            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(self.config.batch_size), actions] = (rewards + (1 - done) * next_q_values * self.config.gamma)

            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        with self.writer.as_default():
            for ep in range(max_episodes):
                done, episode_reward = False, 0

                self.states = np.zeros([self.config.trace_length,  self.config.state_dim])  # Starts with choosing an action from empty states. Uses rolling window size 4
                self.update_states(self.env.reset())

                while not done:
                    action = self.model.get_action(self.states)

                    next_state, reward, done, _ = self.env.step(action)
                    prev_states = self.states

                    self.update_states(next_state)  # Updates the states array removing oldest when adding newest for sliding window
                    self.buffer.add([prev_states, action, reward * 0.01, self.states, done])

                    episode_reward += reward

                if self.buffer.size() >= self.config.batch_size:
                    self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_target()  # target model gets updated AFTER episode, not during like the regular model.

                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)