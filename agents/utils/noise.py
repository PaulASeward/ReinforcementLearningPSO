import numpy as np


class OrnsteinUhlenbeckActionNoise(object):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    :param config.mean: the mean of the noise
    :param config.sigma: the scale of the noise
    :param config.theta: the rate of mean reversion
    :param config.dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
            self,
            config,
            size,
            initial_noise=None,
    ):
        self._theta = config.ou_theta
        self._mu = config.ou_mean
        self._sigma = config.ou_sigma
        self._dt = config.ou_dt

        self._size = size
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros(self._size)
        self.reset()
        super(OrnsteinUhlenbeckActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:
        noise = (
                self.noise_prev
                + self._theta * (self._mu - self.noise_prev) * self._dt
                + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._size)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._size)


class OUActionNoise:
    def __init__(self, config, x_initial=None):
        self.theta = config.ou_theta
        self.mean = np.zeros(1)
        self._sigma = float(config.ou_sigma) * np.ones(1)
        self._dt = config.ou_dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self._dt
                + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self.mean.shape)
        )

        # Store x into x_prev. Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
