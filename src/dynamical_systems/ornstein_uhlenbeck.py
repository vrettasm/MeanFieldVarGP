import numpy as np
from src.dynamical_systems.stochastic_process import StochasticProcess


class OrnsteinUhlenbeck(StochasticProcess):
    """

    Information about the Ornstein - Uhlenbeck process:

    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    __slots__ = ("_sigma", "_theta", "_sigma_inverse")

    def __init__(self, sigma: float, theta: float, r_seed: int = None):
        """
        Default constructor of the Ornstein-Uhlenbeck (OU) object.

        :param sigma: (float) noise diffusion coefficient.

        :param theta: (float) drift model parameter.

        :param r_seed: (int) random seed.
        """

        # Call the constructor of the parent class.
        super().__init__(r_seed=r_seed)

        # Display class info.
        print(" Creating Ornstein-Uhlenbeck process.")

        # Check for the correct type.
        if isinstance(sigma, float):

            # Store the diffusion noise.
            if sigma > 0.0:
                self._sigma = sigma
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

        # Inverse of sigma noise coefficient.
        self._sigma_inverse = 1.0 / sigma

        # Check for the correct type.
        if isinstance(theta, float):

            # Store the drift parameter.
            if theta > 0.0:
                self._theta = theta
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

    # _end_def_

    @property
    def theta(self):
        """
        Accessor method (getter).

        :return: the drift parameter.
        """
        return self._theta
    # _end_def_

    @theta.setter
    def theta(self, new_value: float):
        """
        Accessor method (setter).

        :param new_value: for the drift parameter.

        :return: None.
        """
        # Make sure input is float.
        new_value = float(new_value)

        # Accept only positive values.
        if new_value > 0.0:

            # Make the change.
            self._theta = new_value
        else:
            # Raise an error with a message.
            raise ValueError(f" {self.__class__.__name__}: The drift value"
                             f" {new_value}, should be strictly positive.")
        # _end_if_

    # _end_def_

    @property
    def sigma(self):
        """
        Accessor method (getter).

        :return: the diffusion noise parameter.
        """
        return self._sigma
    # _end_def_

    @sigma.setter
    def sigma(self, new_value: float):
        """
        Accessor method (setter).

        :param new_value: for the noise coefficient.

        :return: None.
        """

        # Make sure input is float.
        new_value = float(new_value)

        # Accept only positive values.
        if new_value > 0.0:

            # Make the change.
            self._sigma = new_value

            # Update the inverse value.
            self._sigma_inverse = 1.0 / self._sigma
        else:
            # Raise an error with a message.
            raise ValueError(f" {self.__class__.__name__}: The sigma value"
                             f" {new_value}, should be strictly positive. ")
        # _end_if_

    # _end_def_

    @property
    def inverse_sigma(self):
        """
        Accessor method (getter).

        :return: the inverse of diffusion noise parameter.
        """
        return self._sigma_inverse
    # _end_def_

    def make_trajectory(self, t0: float, tf: float, dt: float = 0.01, mu: float = 0.0):
        """
        Generates a realizations of the Ornstein-Uhlenbeck (OU)
        dynamical system within a specified time-window [t0-tf].

        :param t0: (float) initial time point.

        :param tf: (float) final time point.

        :param dt: (float) discrete time-step.

        :param mu: (float) default mean value is zero.

        :return: None.
        """

        # Create a time-window with 'dt' time step.
        tk = np.arange(t0, tf+dt, dt, dtype=float)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value X(t=0) = mu.
        x[0] = mu

        # Random variables (notice how the noise scales with the 'dt').
        ek = np.sqrt(self.sigma * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t-1] + self.theta * (mu - x[t-1]) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

# _end_class_
