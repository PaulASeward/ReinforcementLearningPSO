import tensorflow as tf
import gymnasium as gym
import os
from datetime import datetime
from environment.gym_env_discrete import DiscretePsoGymEnv
from agents.utils.policy import ExponentialDecayGreedyEpsilonPolicy
import numpy as np


class BaseAgent:
    def __init__(self, config):
        self.replay_buffer = None
        self.target_model = None
        self.model = None
        self.policy = None
        self.config = config
        self.log_dir = os.path.join(config.log_dir, config.experiment, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.results_logger = None

        self.raw_env = None
        self.env = None

        self.build_environment()
        if config.policy == "ExponentialDecayGreedyEpsilon":
            self.policy = ExponentialDecayGreedyEpsilonPolicy(epsilon_start=config.epsilon_start, epsilon_end=config.epsilon_end, num_steps=config.train_steps, num_actions=config.num_actions)

    def build_environment(self):
        if self.config.use_mock_data:
            self.raw_env = gym.make("MockDiscretePsoGymEnv-v0", config=self.config)
            self.env = self.raw_env
        else:
            self.raw_env = gym.make("DiscretePsoGymEnv-v0", config=self.config)
            self.env = self.raw_env

            return self.env

    def get_q_values(self, state):
        return self.model.get_action_q_values(state)

    def update_model_target_weights(self):
        if not self.config.use_mock_data:
            weights = self.model.model.get_weights()
            self.target_model.model.set_weights(weights)

    def replay_experience(self, experience_length=10):
        if self.replay_buffer.size() < self.config.batch_size:
            return None  # Not enough experience to replay yet.

        losses = []
        if not self.config.use_mock_data:
            for _ in range(experience_length):  # Why size 10?
                states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)
                targets = self.target_model.predict(states)

                next_q_values = self.target_model.predict(next_states).max(axis=1)

                q_values_target = rewards + (1 - done) * self.config.gamma * next_q_values
                targets[range(self.config.batch_size), actions] = q_values_target

                loss = self.model.train(states, targets)
                losses.append(loss)

        return losses

    def get_actions(self):
        print(f"num_actions: {self.config.num_actions}")

        for index, description in enumerate(self.config.actions_descriptions):
            if index + 1 <= self.config.num_actions:
                action_no = str(index + 1)
                print(f"Action #{action_no} Description: {description}")

    def initialize_current_state(self):
        self.policy.reset()
        observation, swarm_info = self.env.reset()
        return np.reshape(observation, (1, self.config.observation_length))

    def update_memory_and_state(self, current_state, action, reward, next_observation, terminal):
        next_state = np.reshape(next_observation, (1, self.config.observation_length))
        # self.replay_buffer.add([current_state, action, reward*self.config.discount_factor, next_state, terminal])
        self.replay_buffer.add([current_state, action, reward, next_state, terminal])
        return next_state

    def train(self):
        with self.writer.as_default():
            for step in range(self.config.train_steps):
                actions, rewards, swarm_observations, terminal = [], [], [], False
                current_state = self.initialize_current_state()

                while not terminal:
                    q_values = self.get_q_values(current_state)
                    action = self.policy.select_action(q_values)
                    next_observation, reward, terminal, swarm_info = self.env.step(action)

                    current_state = self.update_memory_and_state(current_state, action, reward, next_observation, terminal)

                    actions.append(action)
                    rewards.append(reward)
                    swarm_observations.append(swarm_info)

                losses = self.replay_experience()
                self.update_model_target_weights()  # target model gets updated AFTER episode, not during like the regular model.

                self.results_logger.save_log_statements(step=step + 1, actions=actions, rewards=rewards,
                                                        train_loss=losses, epsilon=self.policy.current_epsilon,
                                                        swarm_observations=swarm_observations)
            self.results_logger.print_execution_time()
