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
        # self.action_offset = [self.swarm.config.w, self.swarm.config.c1, self.swarm.config.c2]
        self.action_offset = [
            offset
            for subswarm in self.subswarm_actions
            for offset in subswarm.action_offset
        ]

    def __call__(self, action):
        # Restructure the flattened action from size(config.num_sub_swarms * 3) to size (config.num_sub_swarms, 3)
        # reshaped_arr = action.reshape(3, 5)
        reformatted_action = np.array(action).reshape(self.config.num_sub_swarms, 3)

        # Action should be a dedicated action for each subswarm
        for i, subswarm_action in enumerate(self.subswarm_actions):
            subswarm_action(reformatted_action[i])


class ContinuousActions:
    def __init__(self, swarm, config):
        self.swarm = swarm
        self.config = config

        self.action_names = ['Inertia Param',
                             'Social Param',
                             'Cognitive Param']
        self.action_offset = [self.swarm.config.w, self.swarm.config.c1, self.swarm.config.c2]

    def __call__(self, action):
        """
        :param action: Tuple of 3 values representing the change in inertia, social, and cognitive parameters. Each value should be in the range [-1, 1]
        """
        action_with_offset = np.array(action) + self.action_offset
        self.swarm.w = action_with_offset[0]
        self.swarm.c1 = action_with_offset[1]
        self.swarm.c2 = action_with_offset[2]
