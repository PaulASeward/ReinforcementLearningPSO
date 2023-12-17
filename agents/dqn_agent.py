from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf
from experience_buffer import ExperienceBufferTutorial as ReplayBuffer
from model_networks.dqn_model import DQNModel
from tf_agents.specs import tensor_spec


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.states = np.zeros([self.config.trace_length, self.config.state_dim])
        self.model = DQNModel(config)
        self.target_model = DQNModel(config)

        self.build_environment()
        self.update_target()
        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def get_actions(self):
        action_tensor_spec = tensor_spec.from_spec(self.raw_env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        print(f"num_actions: {num_actions}")

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def replay_experience(self):
        for _ in range(10):  # Why size 10?
            states, actions, rewards, next_states, done = self.buffer.sample(self.config.batch_size)
            targets = self.target_model.predict(states)  # This is likely unnecessary as it ges rewritten

            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(self.config.batch_size), actions] = (rewards + (1 - done) * next_q_values * self.config.gamma)

            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        with self.writer.as_default():
            for ep in range(max_episodes):
                done, episode_reward = False, 0

                self.states = np.zeros([self.config.trace_length,  self.config.state_dim])  # Starts with choosing an action from empty states. Uses rolling window size 4

                observation = self.env.reset()

                while not done:
                    action = self.model.get_action(observation)
                    next_observation, reward, done, _ = self.env.step(action)
                    print("action: ", action)
                    print("next_observation: ", next_observation)
                    print("reward: ", reward)
                    print("done",  done)
                    print("Last Column", _)


                    self.buffer.add([observation, action, reward * 0.01, next_observation, done])
                    episode_reward += reward

                if self.buffer.size() >= self.config.batch_size:
                    self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_target()  # target model gets updated AFTER episode, not during like the regular model.

                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)