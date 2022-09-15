import numpy as np
from numba import njit
from pathlib import Path
from dill import load as dl_load
from numpy import array as array_t
from dynamical_systems.stochastic_process import StochasticProcess


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
def circular_index(i: int, d: int):
    """
    Auxiliary function.

    :param i: index (int) in circular vector.

    :param d: total (int) vector dimensions.

    :return: the indexes [i-2, i-1, i, i+1]
             around a circular set of values
             from 0 to d.
    """

    return [np.mod(i - 2, d),
            np.mod(i - 1, d),
            i,
            np.mod(i + 1, d)]
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

    __slots__ = ("dim_d",)

    def __init__(self, sigma: array_t, theta: array_t,
                 dim_d: int = 40, r_seed: int = None):
        """
        Default constructor of the L96 object.

        :param sigma: (numpy array) noise diffusion coefficient.

        :param theta: (float) drift forcing constant.

        :param r_seed: (int) random seed (default = None).

        :param dim_d: (int) dimensionality of the model (default = 40).
        """

        # Call the constructor of the parent class.
        super().__init__(r_seed=r_seed)

        # Check the number of input dimensions.
        if dim_d < 4:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Insufficient state vector dimensions: {dim_d}")
        # _end_if_

        # Make sure the inputs are numpy arrays.
        sigma = np.asarray(sigma, dtype=float)
        theta = np.asarray(theta, dtype=float)

        # Store the model dimensions.
        self.dim_d = int(dim_d)

        # Check the dimensions of the input.
        if sigma.ndim == 0:
            # Vector (from scalar).
            self.sigma = sigma * np.ones(dim_d)

        elif sigma.ndim == 1:
            # Copy the vector.
            self.sigma = sigma

        elif sigma.ndim == 2:
            # From full Matrix keep only the diagonal.
            self.sigma = sigma.diagonal()

        else:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong input dimensions: {sigma.ndim}")
        # _end_if_

        # Check the dimensionality.
        if len(self.sigma) != dim_d:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong matrix dimensions: {self._sigma.shape}")
        # _end_if_

        # Store the drift parameter.
        self.theta = theta

        # Load the energy functions.
        self._load_functions()

    # _end_def_

    @property
    def inverse_sigma(self):
        """
        Accessor method.

        :return: the inverse of diffusion noise parameter.
        """
        return 1.0 / self.sigma
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
        x0 = self.theta * np.ones(self.dim_d)

        # Initial conditions time step.
        delta_t = 1.0E-3

        # Perturb the middle of the vector by "+dt".
        x0[int(self.dim_d / 2.0)] += delta_t

        # BURN IN:
        for t in range(5000):
            x0 = x0 + _l96(x0, self.theta) * delta_t
        # _end_for_

        # Allocate array.
        x = np.zeros((dim_t, self.dim_d), dtype=float)

        # Start with the new point.
        x[0] = x0

        # Compute the sqrt of noise.
        # NOTE: the scaling with 'dt'.
        ek = np.sqrt(self.sigma * dt)

        # Repeat the vector for the multiplication.
        ek = np.repeat(ek, dim_t).reshape(self.dim_d, dim_t)

        # Multiply with random variables from N(0, 1).
        ek = ek * self.rng.standard_normal((self.dim_d, dim_t))

        # Create the path by solving the SDE iteratively.
        for t in range(1, dim_t):
            x[t] = x[t - 1] + _l96(x[t - 1], self.theta) * dt + ek.T[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def _construct_functions(self, _func_En: callable, _func_dM: callable,
                             _func_dS: callable):
        """
        TBD
        """

        # Make sure to clear everything
        # BEFORE we make the functions.
        self.Esde.clear()
        self.dEsde_dm.clear()
        self.dEsde_ds.clear()

        def _l96_En(t, *args):

            # State vector dimensions.
            D = self.dim_d

            # The first seven variables
            # are the fixed time-points.
            # So we have to exclude them.
            i0 = 7

            # Extract the mean points.
            i1 = i0 + (D * 4)
            mp = np.reshape(args[i0: i1], (D, 4))

            # Extract the variance points.
            i2 = i1 + (D * 3)
            sp = np.reshape(args[i1: i2], (D, 3))

            # Extract the diffusion noise parameters.
            sigma = np.array(args[i2: i2 + D])

            # Extract the drift parameter.
            theta = args[-1]

            # Collect all the function values in this list.
            f_values = []

            # Iterate through all the system dimensions.
            for i in range(D):

                # Get the circular indexes first.
                idx = circular_index(i, D)

                # Pack the input parameters.
                param = [*args[0:i0],
                         *mp[idx, :].flatten(),
                         *sp[idx, :].flatten(),
                         *sigma[idx], theta]

                # Add the function value to the list.
                f_values.append(_func_En(t, *param))
            # _end_for_

            # Return the list.
            return f_values
        # _end_def_

        # Add the energy function.
        self.Esde.append(_l96_En)

        def _l96_dEn_dm(t, *args):

            # State vector dimensions.
            D = self.dim_d

            # The first seven variables
            # are the fixed time-points.
            # So we have to exclude them.
            i0 = 7

            # Extract the mean points.
            i1 = i0 + (D * 4)
            mp = np.reshape(args[i0: i1], (D, 4))

            # Extract the variance points.
            i2 = i1 + (D * 3)
            sp = np.reshape(args[i1: i2], (D, 3))

            # Extract the diffusion noise parameters.
            sigma = np.array(args[i2: i2 + D])

            # Extract the drift parameter.
            theta = args[-1]

            # Collect all the function values in this list.
            f_values = []

            # Iterate through all the system dimensions.
            for i in range(D):

                # Get the circular indexes first.
                idx = circular_index(i, D)

                # Pack the input parameters.
                param = [*args[0:i0],
                         *mp[idx, :].flatten(),
                         *sp[idx, :].flatten(),
                         *sigma[idx], theta]

                # Add the function value to the list.
                f_values.append(_func_dM(t, *param))
            # _end_for_

            # Return the list.
            return f_values
        # _end_def_

        # Add the gradient function.
        self.dEsde_dm.append(_l96_dEn_dm)

        def _l96_dEn_ds(t, *args):

            # State vector dimensions.
            D = self.dim_d

            # The first seven variables
            # are the fixed time-points.
            # So we have to exclude them.
            i0 = 7

            # Extract the mean points.
            i1 = i0 + (D * 4)
            mp = np.reshape(args[i0: i1], (D, 4))

            # Extract the variance points.
            i2 = i1 + (D * 3)
            sp = np.reshape(args[i1: i2], (D, 3))

            # Extract the diffusion noise parameters.
            sigma = np.array(args[i2: i2 + D])

            # Extract the drift parameter.
            theta = args[-1]

            # Collect all the function values in this list.
            f_values = []

            # Iterate through all the system dimensions.
            for i in range(D):

                # Get the circular indexes first.
                idx = circular_index(i, D)

                # Pack the input parameters.
                param = [*args[0:i0],
                         *mp[idx, :].flatten(),
                         *sp[idx, :].flatten(),
                         *sigma[idx], theta]

                # Add the function value to the list.
                f_values.append(_func_dS(t, *param))
            # _end_for_

            # Return the list.
            return f_values
        # _end_def_

        # Add the gradient function.
        self.dEsde_ds.append(_l96_dEn_ds)

    # _end_def_

    def _load_functions(self):
        """
        Auxiliary method that loads the symbolic (lambdafied)
        energy and gradient equations for the Lorenz96 SDE.
        """

        # Initial assignment of functions.
        _func_En, _func_dM, _func_dS = None, None, None

        # Counter of the loaded equations.
        eqn_counter = 0

        # Get the current directory of the file.
        current_dir = Path(__file__).resolve().parent

        # Load the energy file.
        with open(Path(current_dir / "energy_functions/L96_Esde_0.sym"), "rb") as sym_Eqn:

            # Append the energy function.
            _func_En = njit(dl_load(sym_Eqn))

            # Increase by one.
            eqn_counter += 1

        # _end_with_

        # Load the mean-gradient file.
        with open(Path(current_dir / "gradient_functions/dL96_Esde_dM0.sym"), "rb") as sym_Eqn:

            # Append the grad_DM function.
            _func_dM = njit(dl_load(sym_Eqn))

            # Increase by one.
            eqn_counter += 1

        # _end_with_

        # Load the variance-gradient file.
        with open(Path(current_dir / "gradient_functions/dL96_Esde_dS0.sym"), "rb") as sym_Eqn:

            # Append the grad_DS function.
            _func_dS = njit(dl_load(sym_Eqn))

            # Increase by one.
            eqn_counter += 1

        # _end_with_

        # Sanity check.
        if eqn_counter != 3:
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Some symbolic equations failed to load [{eqn_counter}/3].")
        # _end_if_

        # Construct the functions here.
        self._construct_functions(_func_En, _func_dM, _func_dS)

    # _end_def_

# _end_class_
