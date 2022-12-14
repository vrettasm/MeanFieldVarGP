from pathlib import Path

import numpy as np
from dill import load as dl_load
from numba import njit

from dynamical_systems.stochastic_process import StochasticProcess


class DoubleWell(StochasticProcess):
    """
    Information about the double-well potential:

    https://en.wikipedia.org/wiki/Double-well_potential
    """

    def __init__(self, sigma: float, theta: float, r_seed: int = None):
        """
        Default constructor of the DoubleWell (DW) object.

        :param sigma: (float) noise diffusion coefficient.

        :param theta: (float) drift model parameter.

        :param r_seed: (int) random seed.
        """

        # Call the constructor of the parent class.
        super().__init__(r_seed=r_seed)

        # Store the diffusion noise.
        self.sigma = sigma

        # Store the drift parameter.
        self.theta = theta

        # Load the energy functions.
        self._load_functions()
    # _end_def_

    def make_trajectory(self, t0: float, tf: float, dt: float = 0.01):
        """
        Generates a realizations of the double well (DW)
        dynamical system, within a specified time-window.

        :param t0: (float) initial time point.

        :param tf: (float) final time point.

        :param dt: (float) discrete time-step.

        :return: None.
        """

        # Create locally a time-window.
        tk = np.arange(t0, tf+dt, dt, dtype=float)

        # Number of actual time points.
        dim_t = tk.size

        # Preallocate array.
        x = np.zeros(dim_t)

        # The first value is chosen from the "Equilibrium Distribution":
        # This is defined as: x0 = 0.5*N(+mu, K) + 0.5*N(-mu, K)
        x[0] = self.theta

        # Flip the sign with 50% probability.
        if self.rng.random() > 0.5:
            x[0] *= -1.0
        # _end_if_

        # Add Gaussian noise.
        x[0] += np.sqrt(0.5 * self.sigma * dt) * self.rng.standard_normal()

        # Random variables (notice the scale of noise with the 'dt').
        ek = np.sqrt(self.sigma * dt) * self.rng.standard_normal(dim_t)

        # Create the sample path.
        for t in range(1, dim_t):
            x[t] = x[t-1] + \
                   4.0 * x[t-1] * (self.theta - x[t-1] ** 2) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window of inference.
        self.time_window = tk
    # _end_def_

    def _load_functions(self):
        """
        Auxiliary method that loads the symbolic (lambdafied)
        energy and gradient equations for the DoubleWell SDE.
        """

        # Make sure to clear everything BEFORE we load the functions.
        self.Esde.clear()
        self.dEsde_dm.clear()
        self.dEsde_ds.clear()

        # Get the current directory of the file.
        current_dir = Path(__file__).resolve().parent

        # Load the energy file.
        with open(Path(current_dir / "energy_functions/DW_Esde_0.sym"), "rb") as sym_Eqn:

            # Append the energy function.
            self.Esde.append(njit(dl_load(sym_Eqn)))

        # _end_with_

        # Load the mean-gradient file.
        with open(Path(current_dir / "gradient_functions/dDW_Esde_dM0.sym"), "rb") as sym_Eqn:

            # Append the grad_DM function.
            self.dEsde_dm.append(njit(dl_load(sym_Eqn)))

        # _end_with_

        # Load the variance-gradient file.
        with open(Path(current_dir / "gradient_functions/dDW_Esde_dS0.sym"), "rb") as sym_Eqn:

            # Append the grad_DS function.
            self.dEsde_ds.append(njit(dl_load(sym_Eqn)))

        # _end_with_

    # _end_def_

# _end_class_
