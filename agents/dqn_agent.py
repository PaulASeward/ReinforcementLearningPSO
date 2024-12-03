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
