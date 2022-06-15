import numpy as np
from numpy.random import (default_rng,
                          SeedSequence)


class StochasticProcess(object):
    """
    This is a base (parent) class for all the stochastic process models.
    It holds basic information, such as:
        1) the discrete sample path (xt)
        2) the discrete time window (tk)
        3) random number generator,
        4) etc.
    """

    __slots__ = ("xt", "tk", "sp_rng")

    def __init__(self, r_seed=None):
        """
        Default constructor of the SP object.

        :param r_seed: random seed.
        """

        # Create a random number generator.
        if r_seed is None:
            self.sp_rng = default_rng()
        else:
            self.sp_rng = default_rng(SeedSequence(r_seed))
        # _end_if_

        # Sample-path.
        self.xt = None

        # Time-window.
        self.tk = None
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
        return np.abs(self.tk[1] - self.tk[0])
    # _end_def_

    @property
    def rng(self):
        """
        Accessor method.

        :return: the random number generator.
        """
        return self.sp_rng
    # _end_def_

    def collect_obs(self, n_obs, h_mask=None):
        """
        This function collects a number of noise-free observations
        from the discrete sample path (trajectory). If the 'n_obs'
        parameter is integer, we will return 'n_obs' observations
        collected at equidistant points. If it is a list of indexes
        we will return the samples at the exact locations.

        :param n_obs: Observations density (i.e. the number of obs
        per time unit), or list with the indexes that we want to sample.

        :param h_mask: boolean that masks only the observed values.

        :return: observation times / sample values.
        """

        # Sanity check (1): Check if the stochastic process has been created.
        if (self.tk is None) or (self.xt is None):
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Sample path (or time window) have not been created.")
        # _end_def_

        # Sanity check (2): Check if the sample-path and
        # the time-window have equal lengths.
        if len(self.tk) != len(self.xt):
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Sample path and time window do not have the same length.")
        # _end_if_

        # Placeholder.
        obs_t = None

        # Check the type of n_obs.
        if isinstance(n_obs, int):

            # Get the discrete time step.
            dt = np.diff(self.tk)[0]

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

            # Make sure the entries as sorted
            # and unique.
            obs_t = sorted(np.unique(n_obs))

            # Check the length of the lists.
            if len(obs_t) > len(self.tk):
                raise ValueError(f" {self.__class__.__name__}:"
                                 f" Observation density exceeds the number of samples.")
            # _end_if_

            # Make sure everything is int.
            obs_t = list(map(int, obs_t))

        # _end_if_

        # Extract the complete observations (d = D) at times obs_t.
        obs_y = np.take(self.xt, obs_t, axis=0)

        # Check if a mask has been given.
        if h_mask is not None:

            # Here we have (d < D)
            obs_y = obs_y[:, h_mask]
        # _end_if_

        # Observation (times / values).
        return obs_t, obs_y
    # _end_def_

# _end_class_
