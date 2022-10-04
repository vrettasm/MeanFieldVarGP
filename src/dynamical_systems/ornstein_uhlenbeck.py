import numpy as np
from numba import njit
from pathlib import Path
from dill import load as dl_load
from numpy import array as array_t
from dynamical_systems.stochastic_process import StochasticProcess


class OrnsteinUhlenbeck(StochasticProcess):
    """

    Information about the Ornstein - Uhlenbeck process:

    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, sigma: float, theta: array_t, r_seed: int = None):
        """
        Default constructor of the Ornstein-Uhlenbeck (OU) object.

        :param sigma: (float) noise diffusion coefficient (sigma).

        :param theta: (array) drift model parameters (theta, mu).

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
        Generates a realizations of the Ornstein-Uhlenbeck (OU)
        dynamical system within a specified time-window [t0-tf].

        :param t0: (float) initial time point.

        :param tf: (float) final time point.

        :param dt: (float) discrete time-step.

        :return: None.
        """

        # Extract drift parameters.
        theta, mu = self.theta

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
            x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + ek[t]
        # _end_for_

        # Store the sample path (trajectory).
        self.sample_path = x

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def _load_functions(self):
        """
        Auxiliary method that loads the symbolic (lambdafied)
        energy and gradient equations for the Ornstein-Uhlenbeck.
        """

        # Make sure to clear everything BEFORE we load the functions.
        self.Esde.clear()
        self.dEsde_dm.clear()
        self.dEsde_ds.clear()

        # Get the current directory of the file.
        current_dir = Path(__file__).resolve().parent

        # Load the energy file.
        with open(Path(current_dir / "energy_functions/OU_Esde_0.sym"), "rb") as sym_Eqn:

            # Append the energy function.
            self.Esde.append(njit(dl_load(sym_Eqn)))

        # _end_with_

        # Load the mean-gradient file.
        with open(Path(current_dir / "gradient_functions/dOU_Esde_dM0.sym"), "rb") as sym_Eqn:

            # Append the grad_DM function.
            self.dEsde_dm.append(njit(dl_load(sym_Eqn)))

        # _end_with_

        # Load the variance-gradient file.
        with open(Path(current_dir / "gradient_functions/dOU_Esde_dS0.sym"), "rb") as sym_Eqn:

            # Append the grad_DS function.
            self.dEsde_ds.append(njit(dl_load(sym_Eqn)))

        # _end_with_

    # _end_def_

# _end_class_
