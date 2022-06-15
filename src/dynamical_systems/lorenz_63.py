import numpy as np
from numba import njit
from scipy.linalg import cholesky, LinAlgError
from src.dynamical_systems.stochastic_process import StochasticProcess


@njit
def _l63(state, u):
    """
    Lorenz63 function.

    https://en.wikipedia.org/wiki/Lorenz_system

    :param state: (x, y, z).

    :param u: model parameters (sigma=10, rho=28, beta=8/3).

    :return: One step ahead in the equation.
    """

    # Unpack state.
    x, y, z = state

    # Unpack parameters.
    sigma, rho, beta = u

    # Lorenz equations.
    dx = np.array([sigma * (y - x),
                   (rho - z) * x - y,
                   x * y - beta * z])
    # Return dx.
    return dx
# _end_def_


class Lorenz63(StochasticProcess):
    """
    Class that model the Lorenz 3D (1963) dynamical system.

    https://en.wikipedia.org/wiki/Lorenz_system
    """

    __slots__ = ("sigma_", "theta_", "sigma_inv")

    def __init__(self, sigma, theta, r_seed=None):
        """
        Default constructor of the L63 object.

        :param sigma: noise diffusion coefficient.

        :param theta: drift model vector.

        :param r_seed: random seed.
        """
        # Call the constructor of the parent class.
        super().__init__(r_seed, n_dim=3)

        # Display class info.
        print(" Creating Lorenz-63 process.")

        # Make sure the inputs are arrays.
        sigma = np.asarray(sigma, dtype=float)
        theta = np.asarray(theta, dtype=float)

        # Check the dimensions of the input.
        if sigma.ndim == 0:
            # Diagonal matrix (from scalar).
            self.sigma_ = sigma * np.eye(3)

        elif sigma.ndim == 1:
            # Diagonal matrix (from vector).
            self.sigma_ = np.diag(sigma)

        elif sigma.ndim == 2:
            # Full Matrix.
            self.sigma_ = sigma

        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong input dimensions: {sigma.ndim}")
        # _end_if_

        # Check the dimensionality.
        if self.sigma_.shape != (3, 3):
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong matrix dimensions: {self.sigma_.shape}")
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(self.sigma_) > 0.0):

            # Invert the sigma matrix.
            self.sigma_inv = np.linalg.inv(self.sigma_)
        else:
            raise RuntimeError(f" {self.__class__.__name__}: Noise matrix"
                               f" {self.sigma_} is not positive definite.")
        # _end_if_

        # Check the size of the vector.
        if theta.size == 3:

            # Store the drift vector.
            self.theta_ = theta
        else:
            raise RuntimeError(f" {self.__class__.__name__}: Drift vector"
                               f" {self.theta_} is not [3 x 1].")
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
        self.theta_ = np.asarray(new_value, dtype=float)
    # _end_def_

    @property
    def sigma(self):
        """
        Accessor method.

        :return: the system noise parameter.
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
        # Make sure the input is array.
        new_value = np.asarray(new_value, dtype=float)

        # Check the dimensionality.
        if new_value.shape != (3, 3):
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong matrix dimensions: {new_value.shape}.")
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(new_value) > 0.0):
            # Make the change.
            self.sigma_ = new_value

            # Update the inverse matrix.
            self.sigma_inv = np.linalg.inv(self.sigma_)
        else:
            raise RuntimeError(f" {self.__class__.__name__}: Noise matrix"
                               f" {new_value} is not positive definite.")
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
        Generates a realizations of the Lorenz63 (3D)
        dynamical system, within a specified time-window.

        :param t0: initial time point.

        :param tf: final time point.

        :param dt: discrete time-step.

        :return: None.
        """

        # Create locally a time-window.
        tk = np.arange(t0, tf + dt, dt)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Default starting point.
        x0 = np.ones(3)

        # Initial conditions time step.
        delta_t = 1.0e-3

        # BURN IN: This is used to get in the attractor.
        for t in range(5000):
            x0 = x0 + _l63(x0, self.theta_) * delta_t
        # _end_for_

        # Allocate the array.
        x = np.zeros((dim_t, 3))

        # Start with the new point.
        x[0] = x0

        # Compute the Cholesky decomposition of input matrix.
        try:
            ek = cholesky(self.sigma_ * dt)
        except LinAlgError:
            # Show a warning message.
            print(" Warning : Input matrix was not positive definite."
                  " The diagonal elements will be used instead.")

            # If it fails use the diagonal only.
            ek = np.sqrt(np.eye(3) * self.sigma_ * dt)
        # _end_try_

        # Random variables.
        ek = ek.dot(self.rng.standard_normal((3, dim_t))).T

        # Create the path by solving the "stochastic" Diff.Eq. iteratively.
        for t in range(1, dim_t):
            x[t] = x[t-1] + _l63(x[t-1], self.theta_) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk

    # _end_def_

# _end_class_
