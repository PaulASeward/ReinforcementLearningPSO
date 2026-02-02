import os
import copy

import numpy as np
import gymnasium as gym


class RLEnvConfig(object):
    # AGENT PARAMETERS
    num_episodes = 20
    trace_length = 20
    num_swarm_obs_intervals = 10
    swarm_obs_interval_length = 30

    # Defines environment specific config such as action and observation space.
    observation_length = None
    state_shape = None  #  == env.observation_space.shape

    num_actions = None
    action_dimensions = None

    swarm_algorithm = None
    subswarm_action_dim = None
    num_sub_swarms = None
    swarm_size = None  # Should not live here since it is a PSO param

    # Maybe?
    action_space: gym.spaces.Box = None
    observation_space: gym.spaces.Box = None