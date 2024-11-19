import numpy as np
from pso.pso_swarm import PSOSwarm

class PSOMultiSwarm:

    def __init__(self, objective_function, config):
        # PSO Parameters
        self.config = config
        self.swarm_size = config.swarm_size
        self.objective_function = objective_function
        # clone config to avoid changing the original config
        self.sub_swarm_config = config.clone()
        self.sub_swarm_config.swarm_size = config.swarm_size // config.num_sub_swarms
        self.sub_swarm_config.is_sub_swarm = True
        self.dim = config.dim
        self.num_sub_swarms = config.num_sub_swarms

        self.gbest_val = None
        self.gbest_pos = None

        # Initialize the swarm's positions velocities and best solutions
        self._initialize()

    def reinitialize(self):
        for sub_swarm in self.sub_swarms:
            sub_swarm.reinitialize()

        self.update_swarm_valuations_and_bests()

    def _initialize(self):
        self.sub_swarms = [PSOSwarm(self.objective_function, self.sub_swarm_config) for _ in range(self.num_sub_swarms)]

    def update_swarm_valuations_and_bests(self):
        for sub_swarm in self.sub_swarms:
            sub_swarm.update_swarm_valuations_and_bests()

        self.update_gbest()

    def get_observation(self):
        sub_swarm_observations = [sub_swarm.get_observation() for sub_swarm in self.sub_swarms]
        multiswarm_observation = np.concatenate(sub_swarm_observations, axis=0)
        return multiswarm_observation

    def get_current_best_fitness(self):
        return self.gbest_val

    def update_gbest(self):
        best_sub_swarm_idx = np.argmin([sub_swarm.get_current_best_fitness() for sub_swarm in self.sub_swarms])
        self.gbest_val = self.sub_swarms[best_sub_swarm_idx].gbest_val
        self.gbest_pos = self.sub_swarms[best_sub_swarm_idx].gbest_pos

    def optimize(self):
        for sub_swarm in self.sub_swarms:
            sub_swarm.optimize()
        self.update_gbest()






