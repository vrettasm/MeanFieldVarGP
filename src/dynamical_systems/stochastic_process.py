import numpy as np
from numpy.random import default_rng


class StochasticProcess(object):
    """
    This is a base (parent) class for all the stochastic process models.
    It holds basic information, such as:

        1) the discrete sample path (xt)
        2) the discrete time window (tk)
        3) random number generator (rng)
    """

    __slots__ = ("xt", "tk", "_rng", "Esde", "dEsde_dm", "dEsde_ds")

    def __init__(self, r_seed: int = None):
        """
        Default constructor of the StochasticProcess object.

        :param r_seed: random seed (integer).
        """

        # Create a random number generator.
        if r_seed is None:
            self._rng = default_rng()
        else:
            self._rng = default_rng(seed=r_seed)
        # _end_if_

        # Sample-path.
        self.xt = None

        # Time-window.
        self.tk = None

        # Initialize the energy and gradient dictionaries. These will
        # hold the "lambdafied" functions for each dynamical system.
        self.Esde = {}
        self.dEsde_dm = {}
        self.dEsde_ds = {}
    # _end_def_

    @property
    def sample_path(self):
        """
        Accessor method.

        :return: the sample path.
        """

        # Check if the sample path has benn created.
        if self.xt is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Sample path has not been created.")
        # _end_def_

        return self.xt
    # _end_def_

    @sample_path.setter
    def sample_path(self, new_value):
        """
        Accessor method.

        :param new_value: of the sample path (trajectory).

        :return: None.
        """
        self.xt = new_value
    # _end_def_

    @property
    def time_window(self):
        """
        Accessor method.

        :return: the time window of the path.
        """

        # Check if the sample path is created.
        if self.tk is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Time window has not been created.")
        # _end_def_

        return self.tk
    # _end_def_

    @time_window.setter
    def time_window(self, new_value):
        """
        Accessor method.

        :param new_value: of the time window (for inference).

        :return: None.
        """
        self.tk = new_value
    # _end_def_

    @property
    def time_step(self):
        """
        Accessor method.

        :return: the time step of the discrete path.

        NB: We assume the time-window is uniform.
        """

        # Check if the sample path has been created.
        if self.tk is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Time window has not been created.")
        # _end_def_

        # Return the 'dt'.
        return np.abs(np.diff(self.tk)[0])
    # _end_def_

    @property
    def rng(self):
        """
        Accessor method.

        :return: the random number generator.
        """
        return self._rng
    # _end_def_

    def collect_obs(self, n_obs, h_mask=None):
        """
        This function collects a number of noise-free observations
        from the discrete sample path (trajectory). If the 'n_obs'
        parameter is integer, we will return 'n_obs' observations
        collected at equidistant points. If it is a list of indexes
        we will return the samples at the exact locations.

        :param n_obs: Observations density (i.e. the number of obs
        per time unit), or list with the indexes that we want to
        sample.

        :param h_mask: boolean that masks only the observed values.

        :return: observation times / observation values (noise free).
        """

        # Sanity check (1): Check if the stochastic
        # process has been created.
        if (self.tk is None) or (self.xt is None):
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Sample path (or time window) have not been created.")
        # _end_def_

        # Sanity check (2): Check if the sample-path
        # and the time-window have equal lengths.
        if len(self.tk) != len(self.xt):
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Sample path and time window do not have the same length.")
        # _end_if_

        # Placeholder.
        obs_t = None

        # Check the type of n_obs.
        if isinstance(n_obs, int):

            # Get the discrete time step.
            dt = np.abs(np.diff(self.tk)[0])

            # Check if the required number of observations, per
            # time unit, exceeds the available capacity of samples.
            if n_obs > int(1.0 / dt):
                raise ValueError(f" {self.__class__.__name__}:"
                                 f" Observation density exceeds the number of samples.")
            # _end_def_

            # Total number of observations.
            dim_m = int(np.floor(np.abs(self.tk[0] - self.tk[-1]) * n_obs))

            # Number of discrete time points.
            dim_t = self.tk.size

            # Observation indexes.
            idx = np.linspace(0, dim_t, dim_m + 2, dtype=int)

            # Make sure they are unique and sorted.
            obs_t = np.sort(np.unique(idx[1:-1]))

        elif isinstance(n_obs, list):

            # Make sure the entries are sorted and unique.
            obs_t = sorted(np.unique(n_obs))

            # Check the length of the lists.
            if len(obs_t) > len(self.tk):
                raise ValueError(f" {self.__class__.__name__}:"
                                 f" Observation density exceeds the number of samples.")
            # _end_if_

            # Make sure everything is int.
            obs_t = list(map(int, obs_t))

        # _end_if_

        # Extract the full observations (d = D) at times 'obs_t'.
        obs_y = np.take(self.xt, obs_t, axis=0)

        # Check if a mask has been given.
        if h_mask is not None:

            # Here we have (d < D)
            obs_y = obs_y[:, h_mask]
        # _end_if_

        # Observation (times / values).
        return obs_t, obs_y
    # _end_def_

    def energy(self, t, *args):
        """
        Wrapper method. This method wraps the "lambdafied" energy
        function (for each specific dynamical system) and passes
        the output in the numerical quadrature algorithm.

        The first argument 't' is the one that the quadrature is
        based on. All the other parameters are considered fixed
        during the integration.

        Below we can see the lambdafied function signature. The
        parameters must be passed in the exact same order.

        args = [ℎ0, ℎ1, ℎ2, ℎ3, 𝑐0, 𝑐1, 𝑐2,
                𝑑0𝑚0, 𝑑0𝑚1, 𝑑0𝑚2, 𝑑0𝑚3,
                𝑑1𝑚0, 𝑑1𝑚1, 𝑑1𝑚2, 𝑑1𝑚3,
                𝑑2𝑚0, 𝑑2𝑚1, 𝑑2𝑚2, 𝑑2𝑚3,
                ...
                𝑑k𝑚0, 𝑑k𝑚1, 𝑑k𝑚2, 𝑑k𝑚3,
                𝑑0𝑠0, 𝑑0𝑠1, 𝑑0𝑠2,
                𝑑1𝑠0, 𝑑1𝑠1, 𝑑1𝑠2,
                𝑑2𝑠0, 𝑑2𝑠1, 𝑑2𝑠2,
                ...
                𝑑k𝑠0, 𝑑k𝑠1, 𝑑k𝑠2,
                𝑆𝑖𝑔0, 𝑆𝑖𝑔1, 𝑆𝑖𝑔2, ..., 𝑆𝑖𝑔k,
                θ1, θ2, θ3, ..., θN]

        where k = dim_D-1, and N is the number of drift parameters.

        :param t: time variable.

        :param args: list with the rest of the input parameters.

        :return: Esde vector (dim_D,).
        """

        # List of energy values.
        total_energy = []

        # Localise append function.
        total_energy_append = total_energy.append

        # Run through all energy function.
        for En in self.Esde:

            # Get the output of the lambdafied func.
            total_energy_append(En(t, *args))
        # _end_for_

        # Return the list as numpy array.
        return np.array(total_energy, dtype=float)
    # _end_def_

    def grad_mean(self, t, *args):
        """
        Wrapper method. This method wraps the "lambdafied" gradient
        function (for each specific dynamical system) and passes the
        output in the numerical quadrature algorithm.

        The first argument 't' is the one that the quadrature is based on.
        All the other parameters are considered fixed during the integration.

        Below we can see the lambdafied function signature. The parameters
        must be passed in the exact same order.

        args = [ℎ0, ℎ1, ℎ2, ℎ3, 𝑐0, 𝑐1, 𝑐2,
                𝑑0𝑚0, 𝑑0𝑚1, 𝑑0𝑚2, 𝑑0𝑚3,
                𝑑1𝑚0, 𝑑1𝑚1, 𝑑1𝑚2, 𝑑1𝑚3,
                𝑑2𝑚0, 𝑑2𝑚1, 𝑑2𝑚2, 𝑑2𝑚3,
                ...
                𝑑k𝑚0, 𝑑k𝑚1, 𝑑k𝑚2, 𝑑k𝑚3,
                𝑑0𝑠0, 𝑑0𝑠1, 𝑑0𝑠2,
                𝑑1𝑠0, 𝑑1𝑠1, 𝑑1𝑠2,
                𝑑2𝑠0, 𝑑2𝑠1, 𝑑2𝑠2,
                ...
                𝑑k𝑠0, 𝑑k𝑠1, 𝑑k𝑠2,
                𝑆𝑖𝑔0, 𝑆𝑖𝑔1, 𝑆𝑖𝑔2, ..., 𝑆𝑖𝑔k,
                θ1, θ2, θ3, ..., θN]

        where k = D-1, and N is the number of drift parameters.

        :param t: time variable.

        :param args: list with the rest of the input parameters.

        :return: dEsde_dm vector (4*dim_D,).
        """

        # List of energy values.
        total_grad = []

        # Localise append function.
        total_grad_append = total_grad.append

        # Run through all energy function.
        for gm_ in self.dEsde_dm:

            # Get the output of the lambdafied func.
            total_grad_append(gm_(t, *args))
        # _end_for_

        # Return the list as numpy array.
        return np.array(total_grad, dtype=float)
    # _end_def_

    def grad_variance(self, t, *args):
        """
        Wrapper method. This method wraps the "lambdafied" gradient
        function (for each specific dynamical system) and passes the
        output in the numerical quadrature algorithm.

        The first argument 't' is the one that the quadrature is based on.
        All the other parameters are considered fixed during the integration.

        Below we can see the lambdafied function signature. The parameters
        must be passed in the exact same order.

        params = [ℎ0, ℎ1, ℎ2, ℎ3, 𝑐0, 𝑐1, 𝑐2,
                  𝑑0𝑚0, 𝑑0𝑚1, 𝑑0𝑚2, 𝑑0𝑚3,
                  𝑑1𝑚0, 𝑑1𝑚1, 𝑑1𝑚2, 𝑑1𝑚3,
                  𝑑2𝑚0, 𝑑2𝑚1, 𝑑2𝑚2, 𝑑2𝑚3,
                  ...
                  𝑑k𝑚0, 𝑑k𝑚1, 𝑑k𝑚2, 𝑑k𝑚3,
                  𝑑0𝑠0, 𝑑0𝑠1, 𝑑0𝑠2,
                  𝑑1𝑠0, 𝑑1𝑠1, 𝑑1𝑠2,
                  𝑑2𝑠0, 𝑑2𝑠1, 𝑑2𝑠2,
                  ...
                  𝑑k𝑠0, 𝑑k𝑠1, 𝑑k𝑠2,
                  𝑆𝑖𝑔0, 𝑆𝑖𝑔1, 𝑆𝑖𝑔2, ..., 𝑆𝑖𝑔k,
                  θ1, θ2, θ3, ..., θN]

        where k = D-1, and N is the number of drift parameters.

        :param t: time variable.

        :param args: list with the rest of the input parameters.

        :return: dEsde_ds vector (3*dim_D,).
        """

        # List of energy values.
        total_grad = []

        # Localise append function.
        total_grad_append = total_grad.append

        # Run through all energy function.
        for gs_ in self.dEsde_ds:

            # Get the output of the lambdafied func.
            total_grad_append(gs_(t, *args))
        # _end_for_

        # Return the list as numpy array.
        return np.array(total_grad, dtype=float)
    # _end_def_

# _end_class_
