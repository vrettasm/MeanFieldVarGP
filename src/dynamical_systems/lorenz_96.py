import numpy as np
from numba import njit
from pathlib import Path
from dill import load as dl_load

from src.dynamical_systems.stochastic_process import StochasticProcess


@njit
def _l96(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Auxiliary Lorenz 96 model function.

    https://en.wikipedia.org/wiki/Lorenz_96_model

    :param x: state vector (dim_D x 1).

    :param u: additional parameters (theta).

    :return: One step ahead in the equation.
    """

    # Get the shifted vectors.
    fwd_1x = np.roll(x, -1)
    bwd_1x = np.roll(x, +1)
    bwd_2x = np.roll(x, +2)

    # Return one step ahead differential equation.
    return (fwd_1x - bwd_2x) * bwd_1x - x + u
# _end_def_


class Lorenz96(StochasticProcess):
    """
    Class that model the Lorenz (1996) dynamical system.

    https://en.wikipedia.org/wiki/Lorenz_96_model
    """

    __slots__ = ("dim_D",)

    def __init__(self, sigma: np.ndarray, theta: np.ndarray,
                 dim_D: int = 40, r_seed: int = None) -> None:
        """
        Default constructor of the L96 object.

        :param sigma: (numpy array) noise diffusion coefficient.

        :param theta: (float) drift forcing constant.

        :param r_seed: (int) random seed (default = None).

        :param dim_D: (int) dimensionality of the model (default = 40).
        """

        # Call the constructor of the parent class.
        super().__init__(r_seed=r_seed)

        # Check the number of input dimensions.
        if dim_D < 4:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Insufficient state vector dimensions: {dim_D}")
        # _end_if_

        # Make sure the inputs are numpy arrays.
        sigma = np.asarray(sigma, dtype=float)
        theta = np.asarray(theta, dtype=float)

        # Store the model dimensions.
        self.dim_D = int(dim_D)

        # Check the dimensions of the input.
        if sigma.ndim == 0:
            # Vector (from scalar).
            self.sigma = sigma * np.ones(dim_D)

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
        if len(self.sigma) != dim_D:
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Wrong matrix dimensions: {self._sigma.shape}")
        # _end_if_

        # Store the drift parameter.
        self.theta = theta

        # Load the energy functions.
        self._load_functions()
    # _end_def_

    def make_trajectory(self, t0: float, tf: float,
                        dt: float = 0.01) -> None:
        """
        Generates a realizations of the Lorenz96 dynamical system within
        a specified time-window [t0, tf].

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
        x0 = self.theta * np.ones(self.dim_D)

        # Initial conditions time step.
        delta_t = 1.0E-3

        # Perturb the middle of the vector
        # by a small value.
        x0[int(self.dim_D / 2.0)] += 0.01

        # BURN IN: The higher the number of
        # dimensions the more iterations to
        # burn in.
        for t in range(125 * self.dim_D):
            x0 = x0 + _l96(x0, self.theta) * delta_t
        # _end_for_

        # Sanity check.
        if np.any(np.isnan(x0)):
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Invalid initial point: {x0}.")
        # _end_if_

        # Allocate array.
        x = np.zeros((dim_t, self.dim_D), dtype=float)

        # Start with the new point.
        x[0] = x0

        # Compute the sqrt of noise.
        # NOTE: the scaling with 'dt'.
        ek = np.sqrt(self.sigma * dt)

        # Repeat the vector for the multiplication.
        ek = np.repeat(ek, dim_t).reshape(self.dim_D, dim_t)

        # Multiply with random variables from N(0, 1).
        ek = ek * self.rng.standard_normal((self.dim_D, dim_t))

        # Create the path by solving the SDE iteratively.
        for t in range(1, dim_t):

            # Get the one step before 't-1'.
            x_p = _l96(x[t - 1], self.theta)

            # Sanity check.
            if np.any(np.isnan(x_p)):
                raise RuntimeError(f" {self.__class__.__name__}:"
                                   f" Invalid state vector at t={t}."
                                   f" Reduce the value of dt={dt} and try again.")
            # _end_if_

            # Update the current state.
            x[t] = x[t - 1] + x_p * dt + ek.T[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def _construct_functions(self, _func_En: callable, _func_dM: callable,
                             _func_dS: callable):
        """
        This function builds the Energy and Gradients functions for the Lorenz96
        system, for an arbitrary value of system dimensions D.

        Because the system equations involve different indexes (circular values)
        [i-2, i-1, i, i+1], we have to take extra care and put all the equations
        in the right order. Inevitably this code will be slow, but it will allow
        the used to set any value for 'D' and test the algorithms

        :param _func_En: this (callable) function is the Esde expression that we
        have derived from the symbolic code.

        :param _func_dM: this (callable) function is the dEsde_dM expression that
        we have derived from the symbolic code.

        :param _func_dS: this (callable) function is the dEsde_dS expression that
        we have derived from the symbolic code.

        NOTE: this function must be called after the symbolic equations have been
        loaded.

        :return: None.
        """

        # Make sure to clear everything BEFORE we make the functions.
        self.Esde.clear()
        self.dEsde_dm.clear()
        self.dEsde_ds.clear()

        # State vector dimensions.
        D = self.dim_D

        @njit(inline="always")
        def _circular_index(i: int) -> tuple:
            """
            Auxiliary function.

            :param i: index (int) in circular vector.

            :return: the indexes (i, i+1, i-1, i-2)
                     around a circular set of values
                     from 0 to D-1.
            """
            return (i,
                    np.mod(i + 1, D),
                    np.mod(i - 1, D),
                    np.mod(i - 2, D))
        # _end_def_

        @njit(inline="always")
        def _unpack_args(args: np.ndarray):
            """
            Local function that unpacks the list of arguments.
            Compiled with numba for faster execution.
            """
            # The first seven variables are the fixed time-points.
            i0 = 7

            # Time related variables.
            time_vars = args[0:i0]

            # Extract the mean points.
            i1 = i0 + (D * 4)
            mp = args[i0: i1].reshape(D, 4)

            # Extract the variance points.
            i2 = i1 + (D * 3)
            sp = args[i1: i2].reshape(D, 3)

            # Extract the diffusion noise parameters.
            sigma = args[i2: i2 + D]

            # Extract the drift parameter.
            theta = args[-1]

            # Return the unpacked parameters.
            return time_vars, mp, sp, sigma, theta
        # _end_def_

        def _l96_En(t: float, *args):

            # Unpack the arguments.
            time_vars, mp, sp, sigma, theta = _unpack_args(np.array(args))

            # Collect all the function values in this array.
            f_values = np.zeros(D, dtype=float)

            # Get ALL the circular indexes first.
            circ_idx = np.array(_circular_index(np.arange(D)), dtype=int)

            # Iterate through all the system dimensions.
            for i in range(D):

                # Get the indexes for the i-th dimension.
                idx = circ_idx[:, i]

                # Pack the input parameters.
                param = (*time_vars,
                         *mp[idx, :].ravel(),
                         *sp[idx, :].ravel(),
                         *sigma[idx], theta)

                # Add the function value to the list.
                f_values[i] = _func_En(t, *param)
            # _end_for_

            # Return the array.
            return f_values
        # _end_def_

        # Add the energy function.
        self.Esde.append(_l96_En)

        def _l96_dEn_dm(t: float, *args):

            # Unpack the arguments.
            time_vars, mp, sp, sigma, theta = _unpack_args(np.array(args))

            # Collect all the gradient values in this array.
            f_values = np.zeros((D, D * 4), dtype=float)

            # Get ALL the circular indexes first.
            circ_idx = np.array(_circular_index(np.arange(D)), dtype=int)

            # Iterate through all the system dimensions.
            for i in range(D):

                # Get the indexes for the i-th dimension.
                idx = circ_idx[:, i]

                # Pack the input parameters.
                param = (*time_vars,
                         *mp[idx, :].ravel(),
                         *sp[idx, :].ravel(),
                         *sigma[idx], theta)

                # Get the list of gradients.
                _grad_dm = _func_dM(t, *param)

                # Unroll the gradients.
                for j, _ix in enumerate(idx):
                    f_values[i, 4*_ix: 4*(_ix + 1)] = _grad_dm[4*j: 4*(j + 1)]
                # _end_for_

            # _end_for_

            # Return the array.
            return f_values
        # _end_def_

        # Add the gradient function.
        self.dEsde_dm.append(_l96_dEn_dm)

        def _l96_dEn_ds(t: float, *args):

            # Unpack the arguments.
            time_vars, mp, sp, sigma, theta = _unpack_args(np.array(args))

            # Collect all the gradient values in this array.
            f_values = np.zeros((D, D * 3), dtype=float)

            # Get ALL the circular indexes first.
            circ_idx = np.array(_circular_index(np.arange(D)), dtype=int)

            # Iterate through all the system dimensions.
            for i in range(D):

                # Get the indexes for the i-th dimension.
                idx = circ_idx[:, i]

                # Pack the input parameters.
                param = (*time_vars,
                         *mp[idx, :].ravel(),
                         *sp[idx, :].ravel(),
                         *sigma[idx], theta)

                # Get the list of gradients.
                _grad_ds = _func_dS(t, *param)

                # Unroll the gradients.
                for j, _ix in enumerate(idx):
                    f_values[i, 3*_ix: 3*(_ix + 1)] = _grad_ds[3*j: 3*(j + 1)]
                # _end_for_

            # _end_for_

            # Return the array.
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

        # Get the current directory of the file.
        current_dir = Path(__file__).resolve().parent

        # Load the energy file.
        with open(Path(current_dir / "energy_functions/L96_Esde_0.sym"), "rb") as sym_Eqn:

            # Append the energy function.
            _func_En = njit(dl_load(sym_Eqn))

        # _end_with_

        # Load the mean-gradient file.
        with open(Path(current_dir / "gradient_functions/dL96_Esde_dM0.sym"), "rb") as sym_Eqn:

            # Append the grad_DM function.
            _func_dM = njit(dl_load(sym_Eqn))

        # _end_with_

        # Load the variance-gradient file.
        with open(Path(current_dir / "gradient_functions/dL96_Esde_dS0.sym"), "rb") as sym_Eqn:

            # Append the grad_DS function.
            _func_dS = njit(dl_load(sym_Eqn))

        # _end_with_

        # Construct the functions here.
        self._construct_functions(_func_En, _func_dM, _func_dS)
    # _end_def_

# _end_class_
