import numpy as np


class Reward:
    _penalty_for_negative_reward: float = 0
    _minimum: float = None
    _best_fitness: float | None = None
    _best_relative_fitness_for_reward: float | None = None
    _best_relative_fitness_for_plots: float | None = None
    total_difference: float = 0

    def __init__(self, pso_config, env_config):
        self._penalty_for_negative_reward = env_config.penalty_for_negative_reward
        self._minimum = pso_config.fDeltas[pso_config.func_num - 1]

        self._pso_variance = pso_config.standard_deviations[pso_config.func_num - 1] ** 2
        self._avg_swarm_improvement = pso_config.swarm_improvement_pso[pso_config.func_num - 1]
        self._avg_standard_pso_increase = self._avg_swarm_improvement / env_config.num_episodes

    def __call__(self, current_best_fitness: float, actions_count: int):
        raise NotImplementedError

    def reset(self, current_best_fitness: float):
        self._best_relative_fitness_for_reward = None
        self._best_relative_fitness_for_plots = None
        self.total_difference = 0
        self._best_fitness = current_best_fitness

    def get_fitness_reward_for_plots(self, current_best_fitness: float):
        if self._best_relative_fitness_for_plots is None:
            reward = self._minimum - current_best_fitness
            self._best_relative_fitness_for_plots = current_best_fitness
        else:
            reward = max(self._best_relative_fitness_for_plots - current_best_fitness, 0)  # no penalty in reward
            self._best_relative_fitness_for_plots = min(self._best_relative_fitness_for_plots, current_best_fitness)

        return reward


class MockRandomReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)

    def __call__(self, current_best_fitness: float, actions_count: int):
        return np.random.rand() - 0.5  # Random reward between -0.5 and 0.5


class SimpleReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)

    def __call__(self, current_best_fitness: float, actions_count: int):
        difference = self._best_fitness - current_best_fitness
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_fitness)

        if difference > 0:
            return np.float32(1)
        else:
            return np.float32(self._penalty_for_negative_reward)


class FitnessReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)

    def __call__(self, current_best_fitness: float, actions_count: int):
        difference = self._best_fitness - current_best_fitness
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_fitness)

        if actions_count == 0:
            return np.float32(0)

        if self._best_relative_fitness_for_reward is None:
            reward = self._minimum - current_best_fitness
            self._best_relative_fitness_for_reward = current_best_fitness
        else:
            reward = self._best_relative_fitness_for_reward - current_best_fitness
            self._best_relative_fitness_for_reward = min(self._best_relative_fitness_for_reward, current_best_fitness)

        return max(reward, self._penalty_for_negative_reward)


class RelativeFitnessToPSOReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)

        self.standard_pso_results = np.genfromtxt(pso_config.standard_pso_path, delimiter=',', skip_header=1)
        self.standard_pso_cumulative_relative_fitness = abs(self._minimum - self.standard_pso_results[:, 1])

    def __call__(self, current_best_fitness: float, actions_count: int):
        difference = self._best_fitness - current_best_fitness
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_fitness)

        canonical_distance_from_global_optimum = self.standard_pso_cumulative_relative_fitness[actions_count - 1]

        self._best_relative_fitness_for_reward = current_best_fitness if self._best_relative_fitness_for_reward is None else min(
            self._best_relative_fitness_for_reward, current_best_fitness)

        current_distance_from_global_optimum = abs(self._minimum - self._best_relative_fitness_for_reward)
        reward = canonical_distance_from_global_optimum - current_distance_from_global_optimum
        # reward = max(reward, self._penalty_for_negative_reward)
        # reward /= self._pso_variance

        return reward


class DifferenceReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)

    def __call__(self, current_best_fitness: float, actions_count: int):
        difference = self._best_fitness - current_best_fitness
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_fitness)

        return max(difference, self._penalty_for_negative_reward)


class TotalDifferenceReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)

    def __call__(self, current_best_fitness: float, actions_count: int):
        difference = self._best_fitness - current_best_fitness
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_fitness)

        reward = difference * self.total_difference
        # Clip the reward to the total difference
        if reward > self.total_difference:
            reward = self.total_difference
        return max(reward, self._penalty_for_negative_reward)


class SmoothedTotalDifferenceReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)
        self.standard_start_value = np.genfromtxt(pso_config.standard_pso_path, delimiter=',', skip_header=1)[0, 1]
        self._best_fitness = None
        self._max_episodes = env_config.num_episodes

    def __call__(self, current_best_fitness: float, actions_count: int):
        difference = self._best_fitness - current_best_fitness
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_fitness)

        difference = max(difference, 0)
        if actions_count == 1:
            reward = (self.standard_start_value - self._best_fitness) * self._avg_standard_pso_increase
            # reward = (standard_start_value - self._best_fitness) / self._pso_variance
            return reward
        else:
            current_episode_decimal_percent = actions_count / self._max_episodes
            log_difference = np.log(1 + ((difference * self._avg_standard_pso_increase) / (
                        (max(0.05, 1 - current_episode_decimal_percent)) ** 2)))
            reward = log_difference * self.total_difference

            # Clip the reward to the total difference
            if reward > self.total_difference:
                reward = self.total_difference

            reward /= self._pso_variance

            return max(reward, self._penalty_for_negative_reward)


class NormalizedTotalDifferenceReward(Reward):
    def __init__(self, pso_config, env_config):
        super().__init__(pso_config, env_config)

    def __call__(self, current_best_fitness: float, actions_count: int):
        difference = self._best_fitness - current_best_fitness
        self.total_difference += difference
        self._best_fitness = min(self._best_fitness, current_best_fitness)

        reward = difference * self.total_difference

        # Clip the reward to the total difference
        if reward > self.total_difference:
            reward = self.total_difference

        reward /= self._pso_variance

        return max(reward, self._penalty_for_negative_reward)


reward_functions_mapping = {
    "random_reward": MockRandomReward,
    "simple_reward": SimpleReward,
    "fitness_reward": FitnessReward,
    "relative_fitness_reward": RelativeFitnessToPSOReward,
    "difference_reward": DifferenceReward,
    "total_difference_reward": TotalDifferenceReward,
    "smoothed_total_difference_reward": SmoothedTotalDifferenceReward,
    "normalized_total_difference_reward": NormalizedTotalDifferenceReward,
}
