from agents.agent import BaseAgent
import gymnasium as gym
import numpy as np
from agents.utils.experience_buffer import ExperienceBufferStandard as ReplayBuffer
from agents.model_networks.dqn_model import DQNModel
from utils.logging_utils import DiscreteActionsResultsLogger as ResultsLogger


class DQNAgent(BaseAgent):
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)
        self.states = np.zeros([self.config.trace_length, self.config.observation_length])
        self.model = DQNModel(config)
        self.target_model = DQNModel(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer()

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

    def train2(self):
        with self.writer.as_default():
            for ep in range(self.config.train_steps):
                terminal = False
                actions, rewards = [], []

                self.states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4

                observation, swarm_info = self.env.reset()
                state = np.reshape(observation, (1, self.config.observation_length))
                # observation = observation.observation

                while not terminal:
                    q_values = self.model.get_action_q_values(state)
                    action = self.policy.select_action(q_values)
                    next_observation, reward, terminal, info = self.env.step(action)
                    next_state = np.reshape(next_observation, (1, self.config.observation_length))


                    # reward = reward.numpy()[0]
                    # terminal = bool(1 - done)  # done is 0 (not done) if discount=1.0, and 1 if discount = 0.0

                    self.replay_buffer.add([state, action, reward * self.config.discount_factor, next_state, terminal])

                    actions.append(action)
                    state = next_state
                    rewards.append(reward)


                losses = None
                if not self.config.use_mock_data:
                    if self.replay_buffer.size() >= self.config.batch_size:
                        losses = self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                    self.update_model_target_weights()  # target model gets updated AFTER episode, not during like the regular model.

                self.results_logger.save_log_statements(step=ep+1, actions=actions, rewards=rewards, train_loss=losses, epsilon=self.policy.current_epsilon)
            self.results_logger.print_execution_time()
