import numpy as np


class ContinuousMultiswarmActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config
        self.subswarm_actions = [ContinuousActions(sub_swarm, config) for sub_swarm in self.swarm.sub_swarms]
        self.action_names = [
            f"SubSwarm {i + 1} {action_name}"
            for i, subswarm in enumerate(self.subswarm_actions)
            for action_name in subswarm.action_names
        ]

        self.practical_action_high_limit = [
            limit
            for subswarm in self.subswarm_actions
            for limit in subswarm.practical_action_high_limit
        ]
        self.practical_action_low_limit = [
            limit
            for subswarm in self.subswarm_actions
            for limit in subswarm.practical_action_low_limit
        ]

    def __call__(self, action):
        # Restructure the flattened action from size(config.num_sub_swarms * 3) to size (config.num_sub_swarms, 3)
        # reshaped_arr = action.reshape(3, 5)
        reformatted_action = np.array(action).reshape(self.config.num_sub_swarms, self.config.subswarm_action_dim)

        # Action should be a dedicated action for each subswarm
        for i, subswarm_action in enumerate(self.subswarm_actions):
            subswarm_action(reformatted_action[i])

    def set_limits(self):
        self.config.lower_bound = np.array([
            subswarm.actual_low_limit_action_space
            for subswarm in self.subswarm_actions
        ], dtype=np.float32).flatten()

        self.config.upper_bound = np.array([
            subswarm.actual_high_limit_action_space
            for subswarm in self.subswarm_actions
        ], dtype=np.float32).flatten()


class ContinuousActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config

        self.action_names = ['PBest Replacement Threshold']
        self.practical_action_high_limit = [1.0]
        self.practical_action_low_limit = [0.75]

        self.actual_low_limit_action_space = [self.config.replacement_threshold_min]
        self.actual_high_limit_action_space = [self.config.replacement_threshold_max]

    def __call__(self, action):
        """
        :param action: Tuple of 3 values representing the change in inertia, social, and cognitive parameters. Each value should be in the range [-1, 1]
        """
        actions = np.array(action)

        self.swarm.pbest_replacement_threshold = np.clip(actions[0], self.config.replacement_threshold_min, self.config.replacement_threshold_max)
        # self.swarm.velocity_braking = np.clip(actions[1], self.config.velocity_braking_min, self.config.velocity_braking_max)

    def set_limits(self):
        self.config.lower_bound = np.array(self.actual_low_limit_action_space, dtype=np.float32)
        self.config.upper_bound = np.array(self.actual_high_limit_action_space, dtype=np.float32)

