import numpy as np
from numpy import array as array_t
from src.numerical.utilities import cholesky_inv, log_det


class FreeEnergy(object):

    def __init__(self, tk: array_t, mu0: array_t, tau0: array_t,
                 drift_func: callable, sde_theta: array_t, sde_sigma: array_t,
                 obs_times: array_t, obs_values: array_t, obs_noise: array_t,
                 h_operator: array_t = None):
        """
        TBD ...

        :param tk: (discrete) time-window of inference [t0, tf].

        :param mu0: (prior) mean value(s) at time t=0.

        :param tau0: (prior) variance value(s) at time t=0.

        :param drift_func: (callable) function of the system.

        :param sde_theta: SDE drift model parameters.

        :param sde_sigma: SDE diffusion noise parameters.

        :param obs_times: (discrete) time-window of observation times.

        :param obs_values: observation values (including noise).

        :param obs_noise: observation noise variance.

        :param h_operator: observation operator (default = None).
        """

        # SDE drift function.
        self.drift_fun = drift_func

        # Prior mean (t=0).
        self._mu0 = np.asarray(mu0, dtype=float)

        # Prior variance diagonal elements (t=0).
        self._tau0 = np.asarray(tau0, dtype=float)

        # Make sure the SDE drift and diffusion noise
        # parameters are entered correctly as arrays.
        self.theta = np.asarray(sde_theta, dtype=float)
        self.sigma = np.asarray(sde_sigma, dtype=float)

        # Discrete time-window: Tk=[t0, tf].
        self.tk = np.asarray(tk, dtype=float)

        # Sanity check.
        if not np.all(np.diff(self.tk) > 0.0):
            raise ValueError(f" {self.__class__.__name__}: Time window [t0, tf] is not increasing.")
        # _end_if_

        # Make sure the observation parameters
        # are entered correctly as arrays.
        self.obs_times = np.asarray(obs_times, dtype=float)
        self.obs_noise = np.asarray(obs_noise, dtype=float)
        self.obs_values = np.asarray(obs_values, dtype=float)

        # Check if an observation operator is provided.
        if h_operator is None:

            # In the default case we simply set it to '1'.
            self.h_operator = np.array(1.0)
        else:

            # Here we copy the input array.
            self.h_operator = np.asarray(h_operator, dtype=float)
        # _end_if_

        # Infer the system dimensions.
        self.dim_D = self.sigma.size

        # Infer the observations dimensions.
        if self.obs_values.ndim in (0, 1):

            # In this case the observations
            # are scalars (i.e. d=1).
            self.dim_d = 1

        else:

            # The dimensions of the observations vector.
            self.dim_d = self.obs_values.shape[1]

        # _end_if_

        # Number of observations.
        self.num_M = len(obs_times)

        # Sanity check.
        if self.dim_d > self.dim_D:
            raise RuntimeError(f" {self.__class__.__name__}: System dimensions."
                               f" {self.dim_d} should be less than, or equal, to {self.dim_D}.")
        # _end_if_

        # Sanity check.
        if self.num_M != self.obs_values.shape[0]:
            raise RuntimeError(f" {self.__class__.__name__}:  System dimensions."
                               f" Observation times {self.num_M} should be equal"
                               f" to observation values {self.obs_values.shape[0]}.")
        # _end_if_

        # Number of mean points (for a 3rd order polynomial).
        mean_points = (3 * self.num_M + 4)

        # Total number of mean points (scaled with the dimensions).
        self.total_num_mp = self.dim_D * mean_points

        # Number of variance points (for a 2nd order polynomial).
        vars_points = (2 * self.num_M + 3)

        # Number of variance points (scaled with the dimensions).
        # self.total_num_sp = self.dim_D * vars_points

        # Indexes of the observations at the mean/var points.
        self.Tkm = np.arange(3, mean_points - 1, step=3)
        self.Tks = np.arange(2, vars_points - 1, step=2)

    # _end_def_

    @property
    def prior_mu0(self):
        """
        Accessor method (getter).

        :return: the prior mean at time t=0.
        """
        return self._mu0
    # _end_def_

    @prior_mu0.setter
    def prior_mu0(self, new_value: array_t):
        """
        Accessor method (setter).

        NOTE: This should be used only if we optimize the prior moments.

        :return: None.
        """
        self._mu0 = np.asarray(new_value, dtype=float)
    # _end_def_

    @property
    def prior_tau0(self):
        """
        Accessor method (getter).

        :return: the prior variance at time t=0.
        """
        return self._tau0
    # _end_def_

    @prior_tau0.setter
    def prior_tau0(self, new_value: array_t):
        """
        Accessor method (setter).

        NOTE: This should be used only if we optimize the prior moments.

        :return: None.
        """
        self._tau0 = np.asarray(new_value, dtype=float)

    # _end_def_

    @property
    def sde_theta(self):
        """
        Accessor method (getter).

        :return: the SDE drift function parameters.
        """
        return self.theta
    # _end_def_

    @sde_theta.setter
    def sde_theta(self, new_value: array_t):
        """
        Accessor method (setter).

        NOTE: This should be used only if we optimize the drift parameters.

        :return: None.
        """
        self.theta = np.asarray(new_value, dtype=float)
    # _end_def_

    @property
    def sde_sigma(self):
        """
        Accessor method (getter).

        :return: the SDE diffusion noise coefficient.
        """
        return self.sigma
    # _end_def_

    @sde_sigma.setter
    def sde_sigma(self, new_value: array_t):
        """
        Accessor method (setter).

        NOTE: This should be used only if we optimize the diffusion parameters.

        :return: None.
        """
        self.sigma = np.asarray(new_value, dtype=float)
    # _end_def_

    def E_kl0(self, m0: array_t, s0: array_t):
        """
        Energy of the initial state with a Gaussian
        prior q(x|t=0).

        NOTE: Eq. (15) in the paper.

        :param m0: marginal mean at t=0, (dim_D,).

        :param s0: marginal variance at t=0, (dim_D).

        :return: energy of the initial state E0, (scalar)
        and it derivative with respect ot 'm0' and 's0'.
        """

        # Difference of the two "mean" vectors.
        z0 = m0 - self._mu0

        # Energy of the initial moment.
        # NOTE: This formula works only because the matrices 'tau0' and 's0' are
        # diagonal, hence we consider only their diagonal elements, and we work
        # only with vectors.
        E0 = 0.5 * (np.log(np.prod(self._tau0 / s0)) +
                    np.sum((z0**2 + s0 - self._tau0) / self._tau0))

        # Auxiliary variable.
        one_ = np.ones(1)

        # Compute the gradients. We use the "atleast_1d" to ensure
        # that the scalar cases (even though rare) will be covered.
        dE_dm0 = np.atleast_1d(z0 / self._tau0)
        dE_ds0 = np.atleast_1d(0.5 * (one_ / self._tau0 - one_ / s0))

        # Kullback-Liebler value and its derivative, at t=0.
        return E0, np.concatenate((dE_dm0, dE_ds0), axis=0)

    # _end_def_

    def E_sde(self, mean_pts, vars_pts):

        # Number of discrete intervals
        # (between the observations).
        L = self.obs_times.size - 1

        # Initialize energy for the SDE.
        Esde = 0.0

        # Calculate energy from all 'L' time intervals.
        for n in range(L):

            # Take the limits of the observation's interval.
            ti, tj = self.obs_times[n], self.obs_times[n+1]

            # Distance between the two observations.
            # NOTE: This should not change for equally spaced observations.
            delta_t = np.abs(tj-ti)

            # Mid-point intervals (for the evaluation of the Esde function).
            # NOTE: These should not change for equally spaced observations.
            h = float(delta_t/3.0)
            c = float(delta_t/2.0)

            # Separate variables for efficiency.
            nth_mean_points = mean_pts[:, (3 * n): (3 * n) + 4]
            nth_vars_points = vars_pts[:, (2 * n): (2 * n) + 3]

            # Get the SDE (partial) energy for the n-th interval.
            Esde += self.drift_fun(self.theta, self.sigma, h, c,
                                   nth_mean_points, nth_vars_points)
        # _end_for_

        # Return the total energy (including the correct scaling).
        # NOTE: This is Eq. (27) in the paper.
        return 0.5 * Esde
    # _end_def_

    def E_obs(self, mean_pts, var_pts):

        # Inverted Ri and Cholesky factor Qi.
        Ri, Qi = cholesky_inv(self.obs_noise)

        # Auxiliary quantity no.1.
        Z = Qi.dot(self.obs_values - self.h_operator.dot(mean_pts))

        # Auxiliary quantity no.2.
        W = Ri.diagonal().T.dot(self.h_operator.dot(var_pts))

        # Initialize observations' energy.
        Eobs = 0.0

        # Calculate from all 'M' observations.
        for k in range(self.num_M):

            # Get the auxiliary value at this point.
            Zk = Z[k]

            # Compute the energy of the k-th observation.
            Eobs += Zk.T.dot(Zk) + W[k]
        # _end_for_

        # Logarithm of 2*pi.
        log2pi = 1.8378770664093453

        # Final energy value (including the constants).
        Eobs += self.num_M * (self.dim_d * log2pi + log_det(self.obs_noise))

        # Return the total observation energy and its gradients.
        # NOTE: This is Eq. (17) in the paper.
        return 0.5 * Eobs, ...
    # _end_def_

    def E_total(self, x):

        # Separate the mean from the variance points.
        mean_points = np.reshape(x[0:self.total_num_mp],
                                 (self.dim_D, (3*self.num_M + 4)))

        # The variance points are in log-space to ensure positivity,
        # so we pass them through the exponential function first.
        vars_points = np.reshape(np.exp(x[self.total_num_mp:]),
                                 (self.dim_D, (2*self.num_M + 3)))

        # Return the total (free) energy as the sum of the individual
        # components. NOTE: If we want to optimize the hyperparameter
        # we should add another term, e.g. E_param, and include it in
        # the total sum of energy values.
        return self.E_kl0(mean_points[:, 0], vars_points[:, 0]) +\
               self.E_sde(mean_points, vars_points) +\
               self.E_obs(mean_points[:, self.Tkm], vars_points[:, self.Tks])
    # _end_def_

# _end_class_
