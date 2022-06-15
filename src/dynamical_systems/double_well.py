import numpy as np
from src.dynamical_systems.stochastic_process import StochasticProcess


class DoubleWell(StochasticProcess):
    """
    Information about the double-well potential:

    https://en.wikipedia.org/wiki/Double-well_potential
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
        print(" Creating Double-Well process.")

        # Check for the correct type.
        if isinstance(sigma, float):

            # Check for positivity.
            if sigma > 0.0:

                # Store the diffusion noise.
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

        # Inverse of sigma noise coefficient.
        self.sigma_inv = 1.0 / sigma

        # Check for the correct type.
        if isinstance(theta, float):

            # Store the drift parameter.
            self.theta_ = theta
        else:
            raise TypeError(f" {self.__class__.__name__}:"
                            f" The drift model parameter: {theta},"
                            f" should be floating point number.")
        # _end_if_

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

        # Store the drift parameter.
        self.theta_ = float(new_value)
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

        # Make sure the new_value is float.
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

    def make_trajectory(self, t0, tf, dt=0.01):
        """
        Generates a realizations of the double well (DW)
        dynamical system, within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param dt: discrete time-step.

        :return: None.
        """

        # Create locally a time-window.
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual time points.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value is chosen from the
        #    "Equilibrium Distribution":
        # x0 = 0.5*N(+mu,K) + 0.5*N(-mu,K)
        if self.rng.random() > 0.5:
            x[0] = +self.theta_
        else:
            x[0] = -self.theta_
        # _end_if_

        # Add Gaussian noise.
        x[0] += np.sqrt(0.5 * self.sigma_ * dt) * self.rng.standard_normal()

        # Random variables (notice the scale of noise with the 'dt').
        ek = np.sqrt(self.sigma_ * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t-1] + \
                   4.0 * x[t-1] * (self.theta_ - x[t-1] ** 2) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

# _end_class_
