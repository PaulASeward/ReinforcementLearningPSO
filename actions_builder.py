from environment.actions.continuous_actions import ContinuousActions, ContinuousMultiswarmActions
from environment.actions.actions import Action
from environment.actions.discrete_actions import DiscreteActions, DiscreteMultiswarmActions
from pso.pso_swarm import PSOSwarm
from pso.pso_multiswarm import PSOMultiSwarm
import numpy as np

from environment.actions.discrete_actions_library import reset_slow_particles, reset_all_particles_keep_global_best, reset_all_particles_without_memory_sharing, reshare_information_with_global_swarm
from pso.cec_benchmark_functions import CEC_functions


def build_discrete_action_space(swarm: PSOSwarm) -> Action:
    action_names = [
        'Reset slow particles',
        'Reset all particles with preserved information',
        'Reset all particles without memory',
        'Reshare information with global swarm'
        ]

    action_methods = {
            0: reset_slow_particles,
            1: reset_all_particles_keep_global_best,
            2: reset_all_particles_without_memory_sharing,
            3: reshare_information_with_global_swarm
        }

    # self.action_methods = {
    #     0: self.reset_all_particles_keep_global_best,
    # }

    # self.action_names = ['Do nothing', 'Increase inertia', 'Decrease inertia', 'Increase social factor', 'Decrease social factor', 'Reset Slow Particles', 'Reset All Particles Keep Global Best']
    # self.action_methods = {
    #     0: self.do_nothing,
    #     1: self.increase_inertia,
    #     2: self.decrease_inertia,
    #     3: self.increase_social_factor,
    #     4: self.decrease_social_factor,
    #     5: self.reset_slow_particles,
    #     6: self.reset_all_particles_keep_global_best,
    # }

    # self.action_names = ['Do nothing', 'Increase all velocities', 'Decrease all velocities', 'Increase max velocity', 'Decrease max velocity', 'Speed up slower half', 'Slow down faster half',  'Reset Slower half', 'Reset All Particles Keep Global Best']
    # self.action_methods = {
    #     0: self.do_nothing,
    #     1: self.increase_all_velocities,
    #     2: self.decrease_all_velocities,
    #     3: self.increase_max_velocity,
    #     4: self.decrease_max_velocity,
    #     5: self.increase_velocities_of_slow_velocities,
    #     6: self.decrease_velocities_of_fast_particles,
    #     7: self.reset_slow_particles,
    #     8: self.reset_all_particles_keep_global_best,
    # }

    # self.action_names = [
    #     'Reset slow particles velocity to random keep position',
    #     'Reset fast particles velocity to random keep position',
    #     'Reset slow particles velocity to random reset position',
    #     'Reset slow velocity to zero reset position'
    # ]
    # self.action_methods = {
    #     0: self.reset_slow_particles_velocity_to_random_keep_position,
    #     1: self.reset_fast_particles_velocity_to_random_keep_position,
    #     2: self.reset_slow_particles_velocity_to_random_reset_position,
    #     3: self.reset_slow_velocity_to_zero_reset_position,
    # }

    # self.action_names = ['Reset slow particles velocity to random keep position',
    #                      'Reset fast particles velocity to random keep position',
    #                      'Reset all particles velocity to random keep position',
    #                      'Reset slow particles velocity to random reset position',
    #                      'Reset fast particles velocity to random reset position',
    #                      'Reset all particles velocity to random reset position',
    #                      'Reset slow particles velocity to zero reset position',
    #                      'Reset fast particles velocity to zero reset position',
    #                      'Reset all particles velocity to zero reset position']
    # self.action_methods = {
    #     0: self.reset_slow_particles_velocity_to_random_keep_position,
    #     1: self.reset_all_particles_velocity_to_random_keep_position,
    #     2: self.reset_fast_particles_velocity_to_random_keep_position,
    #     3: self.reset_slow_particles_velocity_to_random_reset_position,
    #     4: self.reset_particles_velocity_to_random_reset_position,
    #     5: self.reset_fast_particles_velocity_to_random_reset_position,
    #     6: self.reset_slow_velocity_to_zero_reset_position,
    #     7: self.reset_particles_velocity_to_zero_reset_position,
    #     8: self.reset_fast_velocity_to_zero_reset_position,
    # }

    reset_action_space = DiscreteActions(
        swarm=swarm,
        action_names=action_names,
        action_methods=action_methods,
    )
    return reset_action_space


def build_discrete_multi_action_space(swarm: PSOMultiSwarm) -> Action:
    action_names = [
        'Reset slow particles',
        'Reset all particles with preserved information',
        'Reset all particles without memory',
        'Reshare information with global swarm'
        ]

    action_methods = {
            0: reset_slow_particles,
            1: reset_all_particles_keep_global_best,
            2: reset_all_particles_without_memory_sharing,
            3: reshare_information_with_global_swarm
        }

    multi_reset_action_space = DiscreteMultiswarmActions(swarm=swarm, action_names=action_names, action_methods=action_methods)
    return multi_reset_action_space


def build_continuous_action_space(swarm: PSOSwarm) -> Action:
    # action_names = ['PBest Distance Threshold', 'Velocity Braking Factor']
    # action_names = ['Inertia', 'Social', 'Cognitive']
    action_names = ['Velocity Scaling Factor']

    def action_callback(actions, swarm: PSOSwarm, practical_action_low_limit, practical_action_high_limit):
        swarm.abs_max_velocity = np.clip(actions[0], practical_action_low_limit[0], practical_action_high_limit[0])
        # self.swarm.w = np.clip(actions[0], self.practical_action_low_limit[0], self.practical_action_high_limit[0])
        # self.swarm.c1 = np.clip(actions[1], self.practical_action_low_limit[1], self.practical_action_high_limit[1])
        # self.swarm.c2 = np.clip(actions[2], self.practical_action_low_limit[2], self.practical_action_high_limit[2])

        # self.swarm.pbest_replacement_threshold = np.clip(actions[0], self.practical_action_low_limit[0], self.practical_action_high_limit[0])
        # self.swarm.distance_threshold = np.clip(actions[0], self.practical_action_low_limit[0], self.practical_action_high_limit[0])
        # self.swarm.velocity_braking = np.clip(actions[1], self.practical_action_low_limit[1], self.practical_action_high_limit[1])
        # self.swarm.distance_threshold = np.clip(actions[0], 0, self.swarm.pso_config.distance_threshold_max)
        # self.swarm.velocity_scaling_factor = np.clip(actions[1], self.practical_action_low_limit[1], self.practical_action_high_limit[1])
        #
        # if actions[0] > 0.50:
        #     self.reset_all_particles_keep_global_best()

    velocity_action_space = ContinuousActions(
        swarm=swarm,
        action_callback=action_callback,
        action_names=action_names,
        practical_action_low_limit=[10],
        practical_action_high_limit=[190],
        actual_action_low_limit=[10],
        actual_action_high_limit=[190],
    )

    # velocity_action_space.actual_low_limit_action_space = [10]
    # velocity_action_space.actual_high_limit_action_space = [190]
    # velocity_action_space.actual_low_limit_action_space = [0.75]
    # velocity_action_space.actual_high_limit_action_space = [1.25]
    # velocity_action_space.actual_low_limit_action_space = [swarm.pso_config.replacement_threshold_min]
    # velocity_action_space.actual_high_limit_action_space = [swarm.pso_config.replacement_threshold_max]

    # velocity_action_space.actual_low_limit_action_space = [swarm.pso_config.w_min, swarm.pso_config.c_min, swarm.pso_config.c_min]
    # velocity_action_space.actual_high_limit_action_space = [swarm.pso_config.w_max, swarm.pso_config.c_max, swarm.pso_config.c_max]
    # velocity_action_space.practical_action_low_limit = [swarm.pso_config.w_min, swarm.pso_config.c_min, swarm.pso_config.c_min]
    # velocity_action_space.practical_action_high_limit = [swarm.pso_config.w_max, swarm.pso_config.c_max, swarm.pso_config.c_max]

    # velocity_action_space.practical_action_low_limit = [0, self.swarm.pso_config.velocity_braking_min]
    # velocity_action_space.practical_action_high_limit = [self.swarm.pso_config.distance_threshold_max, self.swarm.pso_config.velocity_braking_max]
    #
    # velocity_action_space.actual_low_limit_action_space = [self.swarm.pso_config.distance_threshold_min, self.swarm.pso_config.velocity_braking_min]
    # velocity_action_space.actual_high_limit_action_space = [self.swarm.pso_config.distance_threshold_max, self.swarm.pso_config.velocity_braking_max]

    return velocity_action_space


def build_continuous_multiswarm_action_space(swarm: PSOMultiSwarm) -> Action:
    action_names = ['Velocity Scaling Factor']

    def action_callback(actions, swarm: PSOSwarm, practical_action_low_limit, practical_action_high_limit):
        swarm.abs_max_velocity = np.clip(actions[0], practical_action_low_limit[0], practical_action_high_limit[0])

    velocity_action_space = ContinuousMultiswarmActions(
        swarm=swarm,
        action_callback=action_callback,
        action_names=action_names,
        practical_action_low_limit=[10],
        practical_action_high_limit=[190],
        actual_action_low_limit=[10],
        actual_action_high_limit=[190],
    )
    return velocity_action_space



