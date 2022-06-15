import numpy as np
from src.dynamical_systems.stochastic_process import StochasticProcess


class OrnsteinUhlenbeck(StochasticProcess):
    """

    Information about the Ornstein - Uhlenbeck process:

    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    __slots__ = ("sigma_", "theta_", "sigma_inv")

    def __init__(self, sigma, theta, r_seed=None):
        """
        Default constructor of the DW object.

        :param sigma: noise diffusion coefficient.

        :param theta: drift model parameter.

        :param r_seed: random seed.
        """
        # Call the constructor of the parent class.
        super().__init__(r_seed, n_dim=1)

        # Display class info.
        print(" Creating Ornstein-Uhlenbeck process.")

        # Check for the correct type.
        if isinstance(sigma, float):

            # Store the diffusion noise.
            if sigma > 0.0:
                self.sigma_ = sigma
            else:
                raise ValueError(f" {self.__class__.__name__}:"
                                 f" The diffusion noise value: {sigma},"
                                 f" should be strictly positive.")
            # _end_if_
        else:
            raise TypeError(f" {self.__class__.__name__}:"
                            f" The diffusion noise value: {sigma},"
                            f" should be floating point number.")
        # _end_if_

        # Check for the correct type.
        if isinstance(theta, float):

            # Store the drift parameter.
            if theta > 0.0:
                self.theta_ = theta
            else:
                raise ValueError(f" {self.__class__.__name__}:"
                                 f" The drift parameter: {theta},"
                                 f" should be strictly positive.")
            # _end_if_
        else:
            raise TypeError(f" {self.__class__.__name__}:"
                            f" The drift model parameter: {theta},"
                            f" should be floating point number.")
        # _end_if_

        # Inverse of sigma noise.
        self.sigma_inv = 1.0 / sigma
    # _end_def_

    @property
    def theta(self):
        """
        Accessor method.

        :return: the drift parameter.
        """
        return self.theta_
    # _end_def_

    @theta.setter
    def theta(self, new_value):
        """
        Accessor method.

        :param new_value: for the drift parameter.

        :return: None.
        """
        # Make sure input is float.
        new_value = float(new_value)

        # Accept only positive values.
        if new_value > 0.0:

            # Make the change.
            self.theta_ = new_value
        else:
            # Raise an error with a message.
            raise ValueError(f" {self.__class__.__name__}: The drift value"
                             f" {new_value}, should be strictly positive.")
        # _end_if_

    # _end_def_

    @property
    def sigma(self):
        """
        Accessor method.

        :return: the diffusion noise parameter.
        """
        return self.sigma_
    # _end_def_

    @sigma.setter
    def sigma(self, new_value):
        """
        Accessor method.

        :param new_value: for the sigma diffusion.

        :return: None.
        """
        # Make sure input is float.
        new_value = float(new_value)

        # Accept only positive values.
        if new_value > 0.0:

            # Make the change.
            self.sigma_ = new_value

            # Update the inverse value.
            self.sigma_inv = 1.0 / self.sigma_
        else:
            # Raise an error with a message.
            raise ValueError(f" {self.__class__.__name__}: The sigma value"
                             f" {new_value}, should be strictly positive. ")
        # _end_if_

    # _end_def_

    @property
    def inverse_sigma(self):
        """
        Accessor method.

        :return: the inverse of diffusion noise parameter.
        """
        return self.sigma_inv
    # _end_def_

    def make_trajectory(self, t0, tf, dt=0.01, mu=0.0):
        """
        Generates a realizations of the Ornstein - Uhlenbeck
        (OU) dynamical system, within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param dt: discrete time-step.

        :param mu: default mean value is zero.

        :return: None.
        """

        # Create a time-window.
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value X(t=0) = 0 or X(t=0) ~ N(mu,K)
        x[0] = mu

        # Random variables (notice the scale of noise with the 'dt').
        ek = np.sqrt(self.sigma_ * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t-1] + self.theta_ * (mu - x[t-1]) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

# _end_class_
