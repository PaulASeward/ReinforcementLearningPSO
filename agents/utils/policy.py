"""RL Policy classes."""

import numpy as np


class Policy:
    """
    Policies are used by the agent to choose actions.
    """
    num_actions = None

    def select_action(self, **kwargs):
        """Used by agents to select actions.

        Returns
        -------
        Action index in range [0, num_actions)

        """
        raise NotImplementedError('This method should be override.')

    def get_config(self):
        return {'num_actions': self.num_actions}


class UniformRandomPolicy(Policy):
    """
    Chooses a discrete action with uniform random probability.
    """
    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, **kwargs):
        return np.random.randint(0, self.num_actions)


class GreedyPolicy(Policy):
    """Always returns best action according to Q-values. This is a pure exploitation policy.
    """

    def select_action(self, q_values, **kwargs):
        """
        Parameters
        ----------
        q_values: (array-like)   Array-like structure of floats representing the Q-values for each action.
        """
        return np.argmax(q_values)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.
    Standard greedy-epsilon implementation. With probability epsilon
    choose a random action. Otherwise, choose the greedy action.

    Parameters
    ----------
    epsilon: float  Initial probability of choosing a random action. Can be changed over time.
    """
    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select_action(self, q_values, **kwargs):
        """Run Greedy-Epsilon for the given Q-values.

        Parameters
        ----------
        q_values: (array-like)   Array-like structure of floats representing the Q-values for each action.
        """
        num_actions = q_values.shape[0]
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, num_actions)
        else:
            return np.argmax(q_values)


class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    epsilon_start: int, float
      The initial value at the start of the decay.
    epsilon_end: int, float
      The value of the policy at the end of the decay. Also the minimum value.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self, epsilon_start, epsilon_end, num_steps, num_actions):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_actions = num_actions

        self.decay_rate = float(epsilon_end - epsilon_start) / num_steps
        self.step = 0

    def select_action(self, q_values, **kwargs):
        """Decay epsilon and select action.

        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        """
        epsilon = self.epsilon_start
        epsilon += self.decay_rate * self.step
        epsilon = max(epsilon, self.epsilon_end)

        self.step += 1
        num_actions = q_values.shape[0]

        if np.random.rand() < epsilon:
            return np.random.randint(0, num_actions)
        else:
            return np.argmax(q_values)

    def reset(self):
        """Start the decay over at the start value."""
        self.step = 0


class ExponentialDecayGreedyEpsilonPolicy(Policy):
    """ Policy with a parameter that decays exponentially.

        Like GreedyEpsilonPolicy but the epsilon decays from a start value
        to an end value over k steps.

        Parameters
        ----------
        epsilon_start: int, float
          The initial value of the parameter
        epsilon_end: int, float
          The value of the policy at the end of the decay.
        num_steps: int
          The number of steps over which to decay the value.

        """

    def __init__(self, epsilon_start, epsilon_end, num_steps, num_actions):
        self.current_epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_actions = num_actions

        self.decay_rate = 4 * float(epsilon_start - epsilon_end) / num_steps
        self.step = 0

    def select_action(self, q_values, **kwargs):
        """Decay parameter and select action.

        Parameters
        ----------
        q_values: np.array
          The Q-values for each action.
        """

        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.decay_rate * self.step)
        epsilon = max(self.current_epsilon, self.epsilon_end)

        self.step += 1
        num_actions = q_values.shape[0]

        if np.random.rand() < epsilon:
            return np.random.randint(0, num_actions)
        else:
            return np.argmax(q_values)

    def reset(self):
        """Start the decay over at the start value."""
        self.step = 0


class OrnsteinUhlenbeckActionNoisePolicy(Policy):
    def __init__(self, num_actions, noise, lower_bound, upper_bound):
        self.num_actions = num_actions
        self.noise = noise
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def select_action(self, q_values, **kwargs):
        action = np.clip(q_values + self.noise(), self.lower_bound, self.upper_bound)
        return action

    def reset(self):
        self.noise.reset()

