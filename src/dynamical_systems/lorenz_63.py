import numpy as np
from numba import njit
from pathlib import Path
from dill import load as dl_load
from numpy import array as array_t
from src.dynamical_systems.stochastic_process import StochasticProcess


@njit
def _l63(state: array_t, theta: array_t) -> array_t:
    """
    Lorenz63 auxiliary function.

    https://en.wikipedia.org/wiki/Lorenz_system

    :param state: (x, y, z).

    :param theta: model parameters (sigma=10, rho=28, beta=8/3).

    :return: One step ahead in the equation.
    """

    # Unpack state.
    x, y, z = state

    # Unpack parameters.
    sigma, rho, beta = theta

    # Lorenz equations.
    d_state = array_t([sigma * (y - x),
                       (rho - z) * x - y,
                       x * y - beta * z])
    # Return d_state/dt.
    return d_state
# _end_def_


class Lorenz63(StochasticProcess):
    """
    Information about the Lorenz 3D (1963) dynamical system:

    https://en.wikipedia.org/wiki/Lorenz_system
    """

    __slots__ = ("_sigma", "_theta", "_sigma_inverse")

    def __init__(self, sigma: array_t, theta: array_t, r_seed: int = None):
        """
        Default constructor of the L63 object.

        :param sigma: (numpy array) noise diffusion coefficient. These
        are diagonal elements (variances) from a full matrix.

        :param theta: (numpy array) drift model vector.

        :param r_seed: (int) random seed.
        """

        # Call the constructor of the parent class.
        super().__init__(r_seed=r_seed)

        # Display class info.
        print(" Creating Lorenz-63 process.")

        # Make sure the inputs are numpy arrays.
        sigma = np.asarray(sigma, dtype=float)
        theta = np.asarray(theta, dtype=float)

        # Check the dimensions of the input.
        if sigma.ndim == 0:
            # Vector (from scalar).
            self._sigma = sigma * np.ones(3)

        elif sigma.ndim == 1:
            # Copy the vector.
            self._sigma = sigma

        elif sigma.ndim == 2:
            # From full Matrix keep only the diagonal.
            self._sigma = sigma.diagonal()

        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong input dimensions: {sigma.ndim}")
        # _end_if_

        # Check for the correct matrix dimensions.
        if len(self._sigma) != 3:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong vector dimensions: {self._sigma.shape}")
        # _end_if_

        # Check for positive definiteness.
        if np.all(self._sigma > 0.0):

            # Invert the sigma coefficients.
            self._sigma_inverse = 1.0 / self._sigma
        else:
            raise RuntimeError(f" {self.__class__.__name__}: Noise matrix"
                               f" {self._sigma} is not positive definite.")
        # _end_if_

        # Check the size of the drift vector.
        if theta.size == 3:

            # Store the drift vector.
            self._theta = theta
        else:
            raise RuntimeError(f" {self.__class__.__name__}: Drift vector"
                               f" {self._theta} is not [3 x 1].")
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
    def theta(self, new_value: array_t):
        """
        Accessor method (setter).

        :param new_value: for the drift parameter.

        :return: None.
        """
        self._theta = np.asarray(new_value, dtype=float)
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

        # Make sure the input is array.
        new_value = np.asarray(new_value, dtype=float)

        # Check the dimensionality.
        if new_value.shape != (3,):
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong vector dimensions: {new_value.shape}.")
        # _end_if_

        # Check for positive definiteness.
        if np.all(new_value > 0.0):
            # Make the change.
            self._sigma = new_value

            # Update the inverse matrix elements.
            self._sigma_inverse = 1.0 / self._sigma
        else:
            raise RuntimeError(f" {self.__class__.__name__}: Noise matrix"
                               f" {new_value} is not positive definite.")
        # _end_if_

    # _end_def_

    @property
    def inverse_sigma(self):
        """
        Accessor method (getter).

        :return: the inverse of diffusion noise vector.
        """
        return self._sigma_inverse

    # _end_def_

    def make_trajectory(self, t0: float, tf: float, dt: float = 0.01):
        """
        Generates a realizations of the Lorenz63 (3D)
        dynamical system, within a specified time-window.

        :param t0: (float) initial time point.

        :param tf: (float) final time point.

        :param dt: (float) discrete time-step.

        :return: None.
        """

        # Create locally a time-window.
        tk = np.arange(t0, tf + dt, dt, dtype=float)

        # Number of actual trajectory samples.
        dim_t = tk.size

        # Default starting point [1.0, 1.0, 1.0].
        x0 = np.ones(3)

        # Initial conditions time step.
        delta_t = 1.0E-3

        # BURN IN: This is used to get the state vector inside
        # the "butterfly" attractor.
        for t in range(5000):
            x0 = x0 + _l63(x0, self._theta) * delta_t
        # _end_for_

        # Allocate the array.
        x = np.zeros((dim_t, 3))

        # Start the sample path with the new point.
        x[0] = x0

        # Compute the sqrt of noise.
        # NOTE: the scaling with 'dt'.
        ek = np.sqrt(self._sigma * dt)

        # Repeat the vector for the multiplication.
        ek = np.repeat(ek, dim_t).reshape(3, dim_t)

        # Multiply with random variables from N(0, 1).
        ek = ek * self.rng.standard_normal((3, dim_t))

        # Create the path by solving the SDE iteratively.
        for t in range(1, dim_t):
            x[t] = x[t-1] + _l63(x[t-1], self._theta) * dt + ek.T[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk

    # _end_def_

    def load_functions(self):
        """
        Auxiliary method that load the symbolic equations for the L3D system.
        """

        # Make sure to clear everything BEFORE we load the functions.
        self.Esde.clear()
        self.dEsde_dm.clear()
        self.dEsde_ds.clear()

        # Counter of the loaded equations.
        eqn_counter = 0

        # Get the current directory of the file.
        current_dir = Path(__file__).resolve().cwd()

        # Load all the dimension equations.
        for i in range(3):

            # Load the energy file.
            with open(Path(current_dir / f"energy_functions/L3D_Esde_{i}.sym"), "rb") as sym_Eqn:

                # Append the energy function.
                self.Esde.append(dl_load(sym_Eqn))

                # Increase by one.
                eqn_counter += 1

            # _end_with_

            # Load the mean-gradient file.
            with open(Path(current_dir / f"gradient_functions/dL3D_Esde_dM{i}.sym"), "rb") as sym_Eqn:

                # Append the grad_DM function.
                self.dEsde_dm.append(dl_load(sym_Eqn))

                # Increase by one.
                eqn_counter += 1

            # _end_with_

            # Load the variance-gradient file.
            with open(Path(current_dir / f"gradient_functions/dL3D_Esde_dS{i}.sym"), "rb") as sym_Eqn:

                # Append the grad_DS function.
                self.dEsde_ds.append(dl_load(sym_Eqn))

                # Increase by one.
                eqn_counter += 1

            # _end_with_

        # _end_for_

        # Sanity check.
        if eqn_counter != 9:
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Some symbolic equations failed to load [{eqn_counter}].")
        # _end_if_

    # _end_def_

# _end_class_
