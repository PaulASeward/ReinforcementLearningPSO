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
        self.sub_swarm_size = config.swarm_size // config.num_sub_swarms
        self.sub_swarm_config.swarm_size = self.sub_swarm_size
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

    def reorganize_swarms(self):
        # Reorganize particles into subswarms, grouped by relative fitness.

        # Call each sub_swarm to return a value of their p_best fitness.
        P_vals = np.array([sub_swarm.P_vals for sub_swarm in self.sub_swarms])

        # Make a one-dimensional array of all p_best values
        pbest_vals_flattened = P_vals.flatten()

        # Generate a list representing the rank of each particles pbest value in the global swarm
        global_swarm_rank_indices = np.argsort(pbest_vals_flattened)

        sub_swarm_rank_indices = global_swarm_rank_indices.reshape((self.num_sub_swarms, self.sub_swarm_size))

        range_per_sub_swarm = [(i*self.sub_swarm_size, (i+1)*self.sub_swarm_size) for i in range(self.num_sub_swarms)]

        # Determine the target subswarm for each particle
        sub_swarm_targets = global_swarm_rank_indices // self.sub_swarm_size

        # Identify particles to be swapped
        particles_to_displace = [set() for _ in range(self.num_sub_swarms)]

        for global_index, target_sub_swarm_idx in enumerate(sub_swarm_targets):
            current_sub_swarm_idx = global_index // self.sub_swarm_size
            current_idx_within_sub_swarm = global_index % self.sub_swarm_size

            if current_sub_swarm_idx != target_sub_swarm_idx:
                # Add particle to the set of swaps
                particle = self.sub_swarms[current_sub_swarm_idx].get_particle(current_idx_within_sub_swarm)
                particles_to_displace[current_sub_swarm_idx].add((
                    particle,
                    current_sub_swarm_idx,
                    current_idx_within_sub_swarm,
                    target_sub_swarm_idx
                ))

        # Start with the first particle to swap in to a target sub_swarm
        incoming_particle_meta_data = None
        for subswarm_particles_to_displace in particles_to_displace:
            if subswarm_particles_to_displace:
                incoming_particle_meta_data = subswarm_particles_to_displace.pop()
                subswarm_particles_to_displace.add(incoming_particle_meta_data)
                break


        # Perform the swaps
        while incoming_particle_meta_data is not None:
            incoming_particle, current_sub_swarm_idx, current_idx_within_sub_swarm, target_sub_swarm_idx = incoming_particle_meta_data

            if len(particles_to_displace[target_sub_swarm_idx]) > 0:

                target_particle, target_sub_swarm_idx, target_current_idx_within_sub_swarm, target_target_sub_swarm_idx = particles_to_displace[target_sub_swarm_idx].pop()
                self.sub_swarms[target_sub_swarm_idx].add_particle(incoming_particle, target_current_idx_within_sub_swarm)

                incoming_particle_meta_data = (target_particle, target_sub_swarm_idx, target_current_idx_within_sub_swarm, target_target_sub_swarm_idx)
            else:
                incoming_particle_meta_data = None





    def swap_particle_out_of_target_subswarm(self, incoming_particle, target_sub_swarm_idx, particle_to_displace_idx):
        displaced_particle = self.sub_swarms[target_sub_swarm_idx].get_particle(particle_to_displace_idx)
        self.sub_swarms[target_sub_swarm_idx].add_particle(incoming_particle, particle_to_displace_idx)

        return displaced_particle


    def optimize(self):
        for sub_swarm in self.sub_swarms:
            sub_swarm.optimize()
        self.update_gbest()
        self.reorganize_swarms()






