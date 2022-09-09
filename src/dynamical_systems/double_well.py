import numpy as np
from pathlib import Path
from dill import load as dl_load
from src.dynamical_systems.stochastic_process import StochasticProcess


class DoubleWell(StochasticProcess):
    """
    Information about the double-well potential:

    https://en.wikipedia.org/wiki/Double-well_potential
    """

    __slots__ = ("_sigma", "_theta", "_sigma_inverse")

    def __init__(self, sigma: float, theta: float, r_seed: int = None):
        """
        Default constructor of the DoubleWell (DW) object.

        :param sigma: (float) noise diffusion coefficient.

        :param theta: (float) drift model parameter.

        :param r_seed: (int) random seed.
        """

        # Call the constructor of the parent class.
        super().__init__(r_seed=r_seed)

        # Display class info.
        print(" Creating Double-Well process.")

        # Check for the correct type.
        if isinstance(sigma, float):

            # Check for positivity.
            if sigma > 0.0:

                # Store the diffusion noise.
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
            self._theta = theta
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

        # Store the drift parameter.
        self._theta = float(new_value)
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

        :param new_value: for the sigma diffusion.

        :return: None.
        """

        # Make sure the new_value is float.
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

        # Store the time window (inference).
        self.time_window = tk
    # _end_def_

    def load_functions(self):
        """
        Auxiliary method that load the symbolic equations for the DW system.
        """

        # Make sure to clear everything BEFORE we load the functions.
        self.Esde.clear()
        self.dEsde_dm.clear()
        self.dEsde_ds.clear()

        # Counter of the loaded files.
        file_counter = 0

        # Get the parent folder of the file.
        parent_dir = Path(__file__).resolve().parent

        # Load the energy file.
        with open(Path(parent_dir / "energy_functions/DW_Esde_0.sym"), "rb") as sym_Eqn:

            # Append the energy function.
            self.Esde.append(dl_load(sym_Eqn))

            # Increase by one.
            file_counter += 1

        # _end_with_

        # Load the mean-gradient file.
        with open(Path(parent_dir / "gradient_functions/dDW_Esde_dM0.sym"), "rb") as sym_Eqn:

            # Append the grad_DM function.
            self.dEsde_dm.append(dl_load(sym_Eqn))

            # Increase by one.
            file_counter += 1

        # _end_with_

        # Load the variance-gradient file.
        with open(Path(parent_dir / "gradient_functions/dDW_Esde_dS0.sym"), "rb") as sym_Eqn:

            # Append the grad_DS function.
            self.dEsde_ds.append(dl_load(sym_Eqn))

            # Increase by one.
            file_counter += 1

        # _end_with_

        # Sanity check.
        if file_counter != 3:
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Some symbolic equations failed to load [{file_counter}].")
        # _end_if_

    # _end_def_

# _end_class_
