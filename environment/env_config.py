import gymnasium as gym
from prompt_toolkit.key_binding.bindings.named_commands import self_insert


class RLEnvConfig(object):
    # AGENT PARAMETERS
    num_episodes = 20
    trace_length = 20

    num_swarm_obs_intervals = 10
    swarm_obs_interval_length = 30
    obs_per_episode = swarm_obs_interval_length / num_swarm_obs_intervals

    # Defines environment specific config such as action and observation space.
    observation_length = None
    state_shape = None  #  == env.observation_space.shape

    num_actions = None
    action_dimensions = None
    subswarm_action_dim = None

    # Maybe?
    action_space: gym.spaces.Box = None
    observation_space: gym.spaces.Box = None

    def __init__(self, num_actions, swarm_size, num_sub_swarms, action_dimensions=None, num_episodes=20):
        # TODO: Make this dynamic to the action/observation space
        num_sub_swarms = num_sub_swarms if num_sub_swarms is not None else 1
        self.observation_length = swarm_size * 3 + (1 * num_sub_swarms) + (1 * num_sub_swarms)

        self.num_actions = num_actions
        self.subswarm_action_dim = action_dimensions
        self.action_dimensions = action_dimensions * num_sub_swarms

        self.num_episodes = num_episodes
        self.trace_length = num_episodes if num_episodes < 20 else 20
