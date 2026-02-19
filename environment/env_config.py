import gymnasium as gym

from environment.actions.actions import Action
from actions_builder import build_continuous_action_space, build_continuous_multiswarm_action_space, build_discrete_action_space, build_discrete_multi_action_space
from pso.pso_config import PSOConfig
from pso.pso_multiswarm import PSOMultiSwarm
from pso.pso_swarm import PSOSwarm


class RLEnvConfig(object):
    # AGENT PARAMETERS
    num_episodes = 20
    trace_length = 20

    num_swarm_obs_intervals = None
    swarm_obs_interval_length = None
    obs_per_episode = None

    actions: Action = None

    # Defines environment specific config such as action and observation space.
    observation_length = None
    state_shape = None  #  == env.observation_space.shape

    num_actions = None
    action_dimensions = None
    use_discrete_env = None
    multiswarm = None

    def __init__(self, swarm: PSOSwarm, network_type, pso_config: PSOConfig, num_episodes=20):
        self.use_discrete_env = True if network_type in ("DQN", "DRQN") else False
        self.multiswarm = True if isinstance(swarm, PSOMultiSwarm) else False

        if self.use_discrete_env and self.multiswarm:
            self.actions = build_discrete_multi_action_space(swarm)
        elif network_type in ("DQN", "DRQN") and not self.multiswarm:
            self.actions = build_discrete_action_space(swarm)
        elif network_type in ("DDPG", "DDRPG", "PPO") and self.multiswarm:
            self.actions = build_continuous_multiswarm_action_space(swarm)
        elif network_type in ("DDPG", "DDRPG", "PPO") and not self.multiswarm:
            self.actions = build_continuous_action_space(swarm)
        else:
            raise ValueError(f"Invalid combination of network type {network_type} and swarm type {type(swarm)}")

        self.observation_length = self.actions.observation_length

        self.num_actions = len(self.actions.action_names)
        self.action_dimensions = self.num_actions

        self.num_swarm_obs_intervals = pso_config.num_swarm_obs_intervals
        self.swarm_obs_interval_length = pso_config.swarm_obs_interval_length
        self.obs_per_episode = self.swarm_obs_interval_length * self.num_swarm_obs_intervals

        self.num_episodes = num_episodes
        self.trace_length = num_episodes if num_episodes < 20 else 20

