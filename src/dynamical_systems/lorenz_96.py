import numpy as np
from numba import njit
from numpy import array as array_t
from scipy.linalg import cholesky, LinAlgError
from src.dynamical_systems.stochastic_process import StochasticProcess


@njit
def fwd_1(x: array_t) -> array_t:
    """
    Auxiliary function.

    :param x: input (state) vector.

    :return: The input vector 'x' shifted forward by one position.
    """

    # Shift forward by one.
    return np.roll(x, -1)
# _end_def_


@njit
def bwd_1(x: array_t) -> array_t:
    """
    Auxiliary function.

    :param x: input (state) vector.

    :return: The input vector 'x' shifted backward by one position.
    """

    # Shift backward by one.
    return np.roll(x, +1)
# _end_def_


@njit
def bwd_2(x: array_t) -> array_t:
    """
    Auxiliary function.

    :param x: input (state) vector.

    :return: The input vector 'x' shifted backward by two positions.
    """

    # Shift backward by two.
    return np.roll(x, +2)
# _end_def


@njit
def shift_vectors(x: array_t) -> array_t:
    """
    Auxiliary function.

    :param x: input (state) vector.

    :return: The input vector 'x' shifted by:
        1) forward  by one,
        2) backward by one,
        3) backward by two.
    """

    # Return ALL the shifted vectors: (-1, +1, +2).
    return np.roll(x, -1), np.roll(x, +1), np.roll(x, +2)
# _end_def_


@njit
def _l96(x: array_t, u: array_t) -> array_t:
    """
    Auxiliary Lorenz 96 model function.

    https://en.wikipedia.org/wiki/Lorenz_96_model

    :param x: state vector (dim_d x 1).

    :param u: additional parameters (theta).

    :return: One step ahead in the equation.
    """

    # Get the shifted values.
    fwd_1x, bwd_1x, bwd_2x = shift_vectors(x)

    # Return one step ahead differential equation.
    return (fwd_1x - bwd_2x) * bwd_1x - x + u
# _end_def_


class Lorenz96(StochasticProcess):
    """
    Class that model the Lorenz 40D (1996) dynamical system.

    https://en.wikipedia.org/wiki/Lorenz_96_model
    """

    __slots__ = ("_sigma", "_theta", "_sigma_inverse", "dim_d")

    def __init__(self, sigma: array_t, theta: array_t,
                 r_seed: int = None, dim_d: int = 40):
        """
        Default constructor of the L96 object.

        :param sigma: (numpy array) noise diffusion coefficient.

        :param theta: (float) drift forcing constant.

        :param r_seed: (int) random seed (default = None).

        :param dim_d: (int) dimensionality of the model (default = 40).
        """

        # Call the constructor of the parent class.
        super().__init__(r_seed=r_seed)

        # Display class info.
        print(f" Creating Lorenz-96 (D={dim_d}) process.")

        # Make sure the inputs are numpy arrays.
        sigma = np.asarray(sigma, dtype=float)
        theta = np.asarray(theta, dtype=float)

        # Check the number of input dimensions.
        if dim_d < 4:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Insufficient state vector dimensions: {dim_d}")
        # _end_if_

        # Store the model dimensions.
        self.dim_d = dim_d

        # Check the dimensions of the input.
        if sigma.ndim == 0:
            # Diagonal matrix (from scalar).
            self._sigma = sigma * np.eye(dim_d)

        elif sigma.ndim == 1:
            # Diagonal matrix (from vector).
            self._sigma = np.diag(sigma)

        elif sigma.ndim == 2:
            # Full matrix.
            self._sigma = sigma

        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong input dimensions: {sigma.ndim}")
        # _end_if_

        # Check the dimensionality.
        if self._sigma.shape != (dim_d, dim_d):
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong matrix dimensions: {self._sigma.shape}")
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(self._sigma) > 0.0):

            # Invert Sigma matrix.
            self._sigma_inverse = np.linalg.inv(self._sigma)
        else:
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Noise matrix {self._sigma} is not positive definite.")
        # _end_if_

        # Store the drift parameter.
        self._theta = theta
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
    def theta(self, new_value: array_t):
        """
        Accessor method (setter).

        :param new_value: for the drift parameter.

        :return: None.
        """
        self._theta = new_value
    # _end_def_

    @property
    def sigma(self):
        """
        Accessor method (getter).

        :return: the system noise parameter.
        """
        return self._sigma
    # _end_def_

    @sigma.setter
    def sigma(self, new_value: array_t):
        """
        Accessor method (setter).

        :param new_value: for the sigma diffusion.

        :return: None.
        """

        # Check the dimensionality.
        if new_value.shape != (self.dim_d, self.dim_d):
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong matrix dimensions: {new_value.shape}")
        # _end_if_

        # Check for positive definiteness.
        if np.all(np.linalg.eigvals(new_value) > 0.0):
            # Make the change.
            self._sigma = new_value

            # Update the inverse matrix.
            self._sigma_inverse = np.linalg.inv(self._sigma)
        else:
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Noise matrix {new_value} is not positive definite.")
        # _end_if_
    # _end_def_

    @property
    def inverse_sigma(self):
        """
        Accessor method.

        :return: the inverse of diffusion noise parameter.
        """
        return self._sigma_inverse
    # _end_def_

    def make_trajectory(self, t0: float, tf: float, dt: float = 0.01):
        """
        Generates a realizations of the Lorenz96 (40D)
        dynamical system, within a specified time-window.

        :param t0: (float) initial time point.

        :param tf: (float) final time point.

        :param dt: (float) discrete time-step.

        :return: None.
        """

        # Create a time-window.
        tk = np.arange(t0, tf + dt, dt, dtype=float)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Default starting point.
        x0 = self._theta * np.ones(self.dim_d)

        # Initial conditions time step.
        delta_t = 1.0E-3

        # Perturb the middle of the vector by "+dt".
        x0[int(self.dim_d / 2.0)] += delta_t

        # BURN IN:
        for t in range(5000):
            x0 = x0 + _l96(x0, self._theta) * delta_t
        # _end_for_

        # Allocate array.
        x = np.zeros((dim_t, self.dim_d))

        # Start with the new point.
        x[0] = x0

        # Compute the Cholesky decomposition of input matrix.
        try:
            # Notice the scaling with 'dt'.
            ek = cholesky(self._sigma * dt)

        except LinAlgError:

            # Show a warning message.
            print(" Warning : The input matrix was not positive definite."
                  " The diagonal elements will be used instead.")

            # If it fails use only the diagonal elements.
            ek = np.sqrt(np.eye(self.dim_d) * self._sigma * dt)
        # _end_try_

        # Random variables.
        ek = ek.dot(self.rng.standard_normal((self.dim_d, dim_t))).T

        # Create the path by solving the SDE iteratively.
        for t in range(1, dim_t):
            x[t] = x[t - 1] + _l96(x[t - 1], self._theta) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

# _end_class_
