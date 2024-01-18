from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf
from experience_buffer import ExperienceBufferTutorial as ReplayBuffer
from model_networks.dqn_model import DQNModel
# from tf_agents.specs import tensor_spec
from logging_utils import ResultsLogger, ComputeDqnReturn


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
        # action_tensor_spec = tensor_spec.from_spec(self.raw_env.action_spec())
        # num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        print(f"num_actions: {5}")

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def replay_experience(self):
        losses = []
        for _ in range(10):  # Why size 10?
            states, actions, rewards, next_states, done = self.buffer.sample(self.config.batch_size)
            targets = self.target_model.predict(states)  # This is likely unnecessary as it ges rewritten

            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(self.config.batch_size), actions] = (rewards + (1 - done) * next_q_values * self.config.gamma)

            loss = self.model.train(states, targets)
            losses.append(loss)

        return losses

    def train(self, max_episodes=10000):
        with self.writer.as_default():
            results = ResultsLogger(self.config, self.env, self.model, ComputeDqnReturn(), max_episodes)

            for ep in range(max_episodes):
                done, episode_reward, actions = False, 0.0, []

                self.states = np.zeros([self.config.trace_length,  self.config.state_dim])  # Starts with choosing an action from empty states. Uses rolling window size 4

                observation = self.env.reset()
                observation = observation.observation

                while not done:
                    action = self.model.get_action(observation)
                    actions.append(action)
                    step_type, reward, discount, next_observation = self.env.step(action)

                    reward = reward.numpy()[0]
                    done = bool(1 - discount)  # done is 0 (not done) if discount=1.0, and 1 if discount = 0.0

                    # TODO: Parameterize the reward discount factor
                    self.buffer.add([observation, action, reward * 0.01, next_observation, done])
                    observation = next_observation
                    episode_reward += reward

                # # Mock Data:
                # actions = [0,1,2,3,4,0,1,2,3,4]
                # losses = [1,2,3,4,5,6,7,8,9,10]

                losses = None
                if self.buffer.size() >= self.config.batch_size:
                    losses = self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_target()  # target model gets updated AFTER episode, not during like the regular model.

                results.save_log_statements(step=ep+1, actions=actions, train_loss=losses)
                print(f"Episode#{ep+1} Cumulative Reward:{episode_reward}")
                # print(f"Actions: {actions}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)

            results.plot_log_statements()
