import tensorflow as tf
import os
from datetime import datetime
from agents.utils.policy import ExponentialDecayGreedyEpsilonPolicy


class BaseAgent:
    def __init__(self, config):
        self.replay_buffer = None
        self.states = None
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
        raise NotImplementedError

    def update_model_target_weights(self):
        if not self.config.use_mock_data:
            weights = self.model.model.get_weights()
            self.target_model.model.set_weights(weights)

    # def update_states(self, next_state):
    #     self.states = np.roll(self.states, -1, axis=0)
    #     self.states[-1] = next_state

    def replay_experience(self, experience_length=10):
        losses = []
        if not self.config.use_mock_data:
            for _ in range(experience_length):  # Why size 10?
                states, actions, rewards, next_states, done = self.replay_buffer.sample(self.config.batch_size)
                targets = self.target_model.predict(states)

                next_q_values = self.target_model.predict(next_states).max(axis=1)
                targets[range(self.config.batch_size), actions] = (
                            rewards + (1 - done) * next_q_values * self.config.gamma)

                loss = self.model.train(states, targets)
                losses.append(loss)

        return losses

    def get_actions(self):
        print(f"num_actions: {self.config.num_actions}")

        for index, description in enumerate(self.config.actions_descriptions):
            if index + 1 <= self.config.num_actions:
                action_no = str(index + 1)
                print(f"Action #{action_no} Description: {description}")
