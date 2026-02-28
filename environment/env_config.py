from typing import List
import numpy as np

from environment.actions.actions import Action
from pso.pso_config import PSOConfig


class RLEnvConfig(object):
    # AGENT PARAMETERS
    num_episodes = 20
    trace_length = 20
    obs_per_episode = None

    penalty_for_negative_reward = 0


    # Defines environment specific config such as action and observation space.
    state_shape = None  #  == env.observation_space.shape

    num_actions = None
    action_dimensions = None
    use_discrete_env = None
    env_name = None

    action_names: List[str] = []
    actions_descriptions: List[str] = []

    append_last_action_to_observation = True
    append_episode_completion_percentage_to_observation = True

    swarm_observation_length: int = None
    observation_length: int = None
    lower_bound: np.ndarray = None
    upper_bound: np.ndarray = None
    practical_low_limit_action_space = []
    practical_high_limit_action_space = []

    def __init__(self, network_type, pso_config: PSOConfig, swarm_observation_length: int, num_episodes=20):
        self.use_discrete_env = True if network_type in ("DQN", "DRQN") else False
        self.env_name = self.get_environment_name(pso_config.use_mock_data)

        self.swarm_observation_length = swarm_observation_length
        self.obs_per_episode = pso_config.swarm_obs_interval_length * pso_config.num_swarm_obs_intervals

        self.num_episodes = num_episodes
        self.trace_length = num_episodes if num_episodes < 20 else 20

    def compute_observation_length(self):
        length = self.swarm_observation_length
        if self.append_last_action_to_observation:
            length += self.action_dimensions
        if self.append_episode_completion_percentage_to_observation:
            length += 1
        return length

    def get_environment_name(self, use_mock_data):
        if self.use_discrete_env and use_mock_data:
            return "MockDiscretePsoGymEnv-v0"
        elif self.use_discrete_env and not use_mock_data:
            return "DiscretePsoGymEnv-v0"
        elif not self.use_discrete_env and use_mock_data:
            return "MockContinuousPsoGymEnv-v0"
        elif not self.use_discrete_env and not use_mock_data:
            return "ContinuousPsoGymEnv-v0"
        raise ValueError()

    def set_action_configs(self, actions: Action):
        self.action_names = actions.action_names
        self.practical_low_limit_action_space = actions.practical_low_limit_action_space
        self.practical_high_limit_action_space = actions.practical_high_limit_action_space
        self.lower_bound = actions.lower_bound
        self.upper_bound = actions.upper_bound

        self.num_actions = len(self.action_names)
        self.action_dimensions = 1 if self.use_discrete_env else self.num_actions  # TODO: Remove redundant variable

        self.observation_length = self.compute_observation_length()



