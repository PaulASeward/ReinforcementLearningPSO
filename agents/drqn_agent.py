from agents.agent import BaseAgent
import copy
import numpy as np
from agents.utils.experience_buffer import ExperienceBufferRecurrent as ReplayBuffer
from agents.model_networks.drqn_model import DRQNModel
from utils.logging_utils import DiscreteActionsResultsLogger as ResultsLogger


class DRQNAgent(BaseAgent):
    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)
        self.episode_states = np.zeros([self.config.trace_length, self.config.observation_length])
        self.model = DRQNModel(config)
        self.target_model = DRQNModel(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer()

    def update_episode_states(self, next_observation):
        self.episode_states = np.roll(self.episode_states, -1, axis=0)
        self.episode_states[-1] = next_observation

    def initialize_current_state(self):
        self.episode_states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4
        observation, swarm_info = self.env.reset()
        self.update_episode_states(observation)
        return np.reshape(self.episode_states, [1, self.config.trace_length, self.config.observation_length])

    def update_memory_and_state(self, current_state, action, reward, next_observation, terminal):
        prev_states = copy.deepcopy(self.episode_states)
        self.update_episode_states(next_observation)  # Updates the states array removing oldest when adding newest for sliding window
        self.replay_buffer.add([prev_states, action, reward * self.config.gamma, self.episode_states, terminal])
        # self.replay_buffer.add([prev_states, action, reward, self.episode_states, terminal])
        return np.reshape(self.episode_states, [1, self.config.trace_length, self.config.observation_length])

