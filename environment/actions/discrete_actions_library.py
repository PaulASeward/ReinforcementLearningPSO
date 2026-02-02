import numpy as np
from pso.pso_swarm import PSOSwarm

from pso.pso_multiswarm import PSOSubSwarm


def do_nothing(swarm: PSOSwarm):
    return


def reshare_information_with_global_swarm(swarm: PSOSwarm):
    if type(swarm) == PSOSubSwarm:
        swarm.share_information_with_global_swarm = True

    swarm.update_superswarm_gbest()


def reset_all_particles_without_memory_sharing(swarm: PSOSwarm):
    swarm.reinitialize()

    if type(swarm) == PSOSubSwarm:
        swarm.share_information_with_global_swarm = False


def reset_all_particles_keep_global_best(swarm: PSOSwarm):
    old_gbest_pos = swarm.P[np.argmin(swarm.P_vals)]
    old_gbest_val = np.min(swarm.P_vals)

    swarm.reinitialize()
    if type(swarm) == PSOSubSwarm:
        swarm.share_information_with_global_swarm = True

    # Keep Previous Solution before resetting.
    if old_gbest_val < swarm.gbest_val:
        swarm.gbest_pos = old_gbest_pos
        swarm.gbest_val = old_gbest_val


def reset_all_particles_keep_personal_best(swarm: PSOSwarm):
    old_pbest_pos = swarm.P

    swarm.reinitialize()

    swarm.P = old_pbest_pos
    swarm.update_swarm_valuations_and_bests()


def reset_slow_particles(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    slow_particles = swarm.velocity_magnitudes < avg_velocity
    replacement_positions = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                              size=(swarm.swarm_size, swarm.dimension))
    replacement_velocities = np.full((swarm.swarm_size, swarm.dimension), 0)
    slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match swarm.X

    swarm.X = np.where(slow_particles_reshaped, replacement_positions, swarm.X)
    swarm.V = np.where(slow_particles_reshaped, replacement_velocities, swarm.V)
    swarm.P = np.where(slow_particles_reshaped, swarm.X, swarm.P)

    swarm.update_swarm_valuations_and_bests()


def _reset_particles_position_using_lattice(swarm: PSOSwarm):
    grid_points = int(np.cbrt(swarm.swarm_size))  # Assuming cubic grid
    lattice = np.linspace(-1 * swarm.rangeF, swarm.rangeF, grid_points)
    positions = np.array(np.meshgrid(*([lattice] * swarm.dimension))).T.reshape(-1, swarm.dimension)
    np.random.shuffle(positions)
    swarm.X[:positions.shape[0]] = positions[:swarm.swarm_size]


def _reset_particles_position_to_random(swarm: PSOSwarm):
    swarm.X = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                     size=(swarm.swarm_size, swarm.dimension))


def _reset_particles_velocity_to_random(swarm: PSOSwarm):
    swarm.V = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                     size=(swarm.swarm_size, swarm.dimension))


def _reset_particles_velocity_to_zero(swarm: PSOSwarm):
    swarm.V = np.full((swarm.swarm_size, swarm.dimension), 0.0)


def _forget_memory(swarm: PSOSwarm):
    swarm.P = swarm.X
    swarm.P_vals = None


def reset_all_particles_velocity_to_random_keep_position(swarm: PSOSwarm):
    _reset_particles_velocity_to_random(swarm)


def reset_slow_particles_velocity_to_random_keep_position(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    slow_particles = swarm.velocity_magnitudes < avg_velocity
    replacement_velocities = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                               size=(swarm.swarm_size, swarm.dimension))
    slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match swarm.X
    swarm.V = np.where(slow_particles_reshaped, replacement_velocities, swarm.V)


def reset_fast_particles_velocity_to_random_keep_position(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    fast_particles = swarm.velocity_magnitudes > avg_velocity
    replacement_velocities = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                               size=(swarm.swarm_size, swarm.dimension))
    fast_particles_reshaped = fast_particles[:, np.newaxis]  # Reshape to match swarm.X
    swarm.V = np.where(fast_particles_reshaped, replacement_velocities, swarm.V)


def reset_particles_velocity_to_random_reset_position(swarm: PSOSwarm):
    _reset_particles_position_to_random(swarm)
    _reset_particles_velocity_to_random(swarm)
    swarm.update_swarm_valuations_and_bests()


def reset_slow_particles_velocity_to_random_reset_position(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    slow_particles = swarm.velocity_magnitudes < avg_velocity
    slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match swarm.X
    replacement_velocities = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                               size=(swarm.swarm_size, swarm.dimension))
    replacement_positions = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                              size=(swarm.swarm_size, swarm.dimension))

    swarm.X = np.where(slow_particles_reshaped, replacement_positions, swarm.X)
    swarm.V = np.where(slow_particles_reshaped, replacement_velocities, swarm.V)
    swarm.P = np.where(slow_particles_reshaped, swarm.X, swarm.P)

    swarm.update_swarm_valuations_and_bests()


def reset_fast_particles_velocity_to_random_reset_position(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    fast_particles = swarm.velocity_magnitudes > avg_velocity
    fast_particles_reshaped = fast_particles[:, np.newaxis]  # Reshape to match swarm.X

    replacement_velocities = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                               size=(swarm.swarm_size, swarm.dimension))
    replacement_positions = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                              size=(swarm.swarm_size, swarm.dimension))

    swarm.X = np.where(fast_particles_reshaped, replacement_positions, swarm.X)
    swarm.V = np.where(fast_particles_reshaped, replacement_velocities, swarm.V)
    swarm.P = np.where(fast_particles_reshaped, swarm.X, swarm.P)

    swarm.update_swarm_valuations_and_bests()


def reset_particles_velocity_to_zero_reset_position(swarm: PSOSwarm):
    _reset_particles_position_to_random(swarm)
    _reset_particles_velocity_to_zero(swarm)
    swarm.update_swarm_valuations_and_bests()


def reset_slow_velocity_to_zero_reset_position(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    slow_particles = swarm.velocity_magnitudes < avg_velocity
    slow_particles_reshaped = slow_particles[:, np.newaxis]

    replacement_velocities = np.full((swarm.swarm_size, swarm.dimension), 0.0)
    replacement_positions = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                              size=(swarm.swarm_size, swarm.dimension))

    swarm.X = np.where(slow_particles_reshaped, replacement_positions, swarm.X)
    swarm.V = np.where(slow_particles_reshaped, replacement_velocities, swarm.V)
    swarm.P = np.where(slow_particles_reshaped, swarm.X, swarm.P)
    swarm.update_swarm_valuations_and_bests()


def reset_fast_velocity_to_zero_reset_position(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    fast_particles = swarm.velocity_magnitudes > avg_velocity
    fast_particles_reshaped = fast_particles[:, np.newaxis]

    replacement_velocities = np.full((swarm.swarm_size, swarm.dimension), 0.0)
    replacement_positions = np.random.uniform(low=-1 * swarm.rangeF, high=swarm.rangeF,
                                              size=(swarm.swarm_size, swarm.dimension))

    swarm.X = np.where(fast_particles_reshaped, replacement_positions, swarm.X)
    swarm.V = np.where(fast_particles_reshaped, replacement_velocities, swarm.V)
    swarm.P = np.where(fast_particles_reshaped, swarm.X, swarm.P)

    swarm.update_swarm_valuations_and_bests()


def increase_all_velocities(swarm: PSOSwarm):
    swarm.velocity_scaling_factor *= 1.10


def decrease_all_velocities(swarm: PSOSwarm):
    swarm.velocity_scaling_factor *= 0.90


def _calculate_average_velocity(swarm: PSOSwarm):
    swarm.update_velocity_maginitude()
    return np.mean(swarm.velocity_magnitudes)


def inject_random_perturbations_to_velocities(swarm, selection_type, factor):
    swarm.perturb_velocities = True
    swarm.perturb_velocity_factor = factor
    swarm.perturb_velocity_particle_selection = selection_type


def inject_small_perturbations_to_slow_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=0, factor=0.05)


def inject_medium_perturbations_to_slow_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=0, factor=0.20)


def inject_large_perturbations_to_slow_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=0, factor=0.50)


def inject_small_perturbations_to_fast_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=1, factor=0.05)


def inject_medium_perturbations_to_fast_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=1, factor=0.20)


def inject_large_perturbations_to_fast_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=1, factor=0.50)


def inject_small_perturbations_to_all_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=2, factor=0.05)


def inject_medium_perturbations_to_all_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=2, factor=0.20)


def inject_large_perturbations_to_all_particles(swarm: PSOSwarm):
    inject_random_perturbations_to_velocities(swarm,selection_type=2, factor=0.50)


def increase_velocities_of_slow_velocities(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    slow_particles = swarm.velocity_magnitudes < avg_velocity
    slow_particles_reshaped = slow_particles[:, np.newaxis]  # Reshape to match swarm.X

    faster_velocities = swarm.V * 1.10
    faster_velocities = np.clip(faster_velocities, -swarm.abs_max_velocity, swarm.abs_max_velocity)
    swarm.V = np.where(slow_particles_reshaped, faster_velocities, swarm.V)


def decrease_velocities_of_fast_particles(swarm: PSOSwarm):
    avg_velocity = _calculate_average_velocity(swarm)
    fast_particles = swarm.velocity_magnitudes > avg_velocity
    fast_particles_reshaped = fast_particles[:, np.newaxis]  # Reshape to match swarm.X

    slower_velocities = swarm.V * 0.90
    swarm.V = np.where(fast_particles_reshaped, slower_velocities, swarm.V)


def increase_max_velocity(swarm: PSOSwarm):
    swarm.abs_max_velocity *= 1.10
    swarm.abs_max_velocity = np.clip(swarm.abs_max_velocity, swarm.config.pso_config.v_min,
                                          swarm.config.pso_config.v_max)


def decrease_max_velocity(swarm: PSOSwarm):
    swarm.abs_max_velocity *= 0.90
    swarm.abs_max_velocity = np.clip(swarm.abs_max_velocity, swarm.config.pso_config.v_min,
                                          swarm.config.pso_config.v_max)


def increase_social_factor(swarm: PSOSwarm):
    swarm.c1 *= 1.10  # Social component
    swarm.c1 = np.clip(swarm.c1, swarm.config.pso_config.c_min, swarm.config.pso_config.c_max)

    swarm.c2 *= 0.90  # Cognitive component
    swarm.c2 = np.clip(swarm.c2, swarm.config.pso_config.c_min, swarm.config.pso_config.c_max)


def increase_inertia(swarm: PSOSwarm):
    swarm.w *= 1.10
    swarm.w = np.clip(swarm.w, swarm.config.pso_config.w_min, swarm.config.pso_config.w_max)


def decrease_inertia(swarm: PSOSwarm):
    swarm.w *= 0.90
    swarm.w = np.clip(swarm.w, swarm.config.pso_config.w_min, swarm.config.pso_config.w_max)


def decrease_social_factor(swarm: PSOSwarm):
    swarm.c1 *= 0.90
    swarm.c1 = np.clip(swarm.c1, swarm.config.pso_config.c_min, swarm.config.pso_config.c_max)

    swarm.c2 *= 1.10
    swarm.c2 = np.clip(swarm.c2, swarm.config.pso_config.c_min, swarm.config.pso_config.c_max)


# Threshold actions to promote exploration vs exploitation
def decrease_pbest_replacement_threshold(swarm: PSOSwarm):
    swarm.pbest_replacement_threshold *= swarm.pbest_replacement_threshold_decay
    swarm.pbest_replacement_threshold = np.clip(swarm.pbest_replacement_threshold,
                                                     swarm.config.pso_config.pbest_replacement_threshold_min,
                                                     swarm.config.pso_config.pbest_replacement_threshold_max)


def increase_pbest_replacement_threshold(swarm: PSOSwarm):
    swarm.pbest_replacement_threshold *= 1.10
    swarm.pbest_replacement_threshold = np.clip(swarm.pbest_replacement_threshold,
                                                     swarm.config.pso_config.pbest_replacement_threshold_min,
                                                     swarm.config.pso_config.pbest_replacement_threshold_max)