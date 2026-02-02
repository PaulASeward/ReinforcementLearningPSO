import os
import copy

import numpy as np
import gymnasium as gym


class RLEnvConfig(object):


    # Defines environment specific config such as action and observation space.
    observation_length = None
    state_shape = None  #  == env.observation_space.shape

    num_actions = None

    # Maybe?
    action_space: gym.spaces.Box = None
    observation_space: gym.spaces.Box = None