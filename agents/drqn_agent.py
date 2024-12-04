from agents.agent import BaseAgent

import numpy as np
from agents.utils.experience_buffer import ExperienceBufferRecurrent as ReplayBuffer
from agents.model_networks.drqn_model import DRQNModel
from utils.logging_utils import DiscreteActionsResultsLogger as ResultsLogger


class DRQNAgent(BaseAgent):
    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)
        self.states = np.zeros([self.config.trace_length, self.config.observation_length])
        self.model = DRQNModel(config)
        self.target_model = DRQNModel(config)

        self.update_model_target_weights()
        self.replay_buffer = ReplayBuffer()

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def train(self):
        with self.writer.as_default():
            for ep in range(self.config.train_steps):
                actions, rewards, swarm_observations, terminal = [], [], [], False

                self.states = np.zeros([self.config.trace_length, self.config.observation_length])  # Starts with choosing an action from empty states. Uses rolling window size 4
                observation, swarm_info = self.env.reset()

                self.update_states(observation)  # Check states array update

                while not terminal:
                    input_state = np.reshape(self.states, [1, self.config.trace_length, self.config.observation_length])
                    q_values = self.model.get_action_q_values(input_state)
                    action = self.policy.select_action(q_values)

                    next_state, reward, terminal, swarm_info = self.env.step(action)
                    # next_state = np.reshape(next_observation, [1, self.config.trace_length, self.config.observation_length])

                    prev_states = self.states
                    self.update_states(next_state)  # Updates the states array removing oldest when adding newest for sliding window
                    self.replay_buffer.add([prev_states, action, reward * self.config.discount_factor, self.states, terminal])

                    actions.append(action)
                    rewards.append(reward)
                    swarm_observations.append(swarm_info)


                losses = None
                if self.replay_buffer.size() >= self.config.batch_size:
                    losses = self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_model_target_weights()  # target model gets updated AFTER episode, not during like the regular model.

                self.results_logger.save_log_statements(step=ep+1, actions=actions, rewards=rewards, train_loss=losses, epsilon=self.policy.current_epsilon, swarm_observations=swarm_observations)
            self.results_logger.print_execution_time()
