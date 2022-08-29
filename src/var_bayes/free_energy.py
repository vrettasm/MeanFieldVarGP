import numpy as np
from numpy import array as array_t
from src.numerical.utilities import cholesky_inv, log_det


class FreeEnergy(object):

    def __init__(self, drift_fun: callable, mu0: array_t, tau0: array_t, sde_theta: array_t, sde_sigma: array_t,
                 tx: array_t, obs_values: array_t, obs_noise: array_t, h_operator: array_t = None):
        # SDE drift function.
        self.drift_fun = drift_fun

        # Prior mean (t=0).
        self._mu0 = np.asarray(mu0, dtype=float)

        # Prior co-variance (t=0).
        self._tau0 = np.asarray(tau0, dtype=float)

        # Make sure the drift and diffusion noise parameters
        # are entered correctly as numpy arrays.
        self.theta = np.asarray(sde_theta, dtype=float)
        self.sigma = np.asarray(sde_sigma, dtype=float)

        # Compute the inverse of the sigma.
        self.sigma_inverse, _ = cholesky_inv(self.sigma)

        # Observation times.
        self.tx = np.asarray(tx, dtype=float)

        # Make sure the observations and their noise parameters
        # are entered correctly as numpy arrays.
        self.obs = np.asarray(obs_values, dtype=float)
        self.noise = np.asarray(obs_noise, dtype=float)

        # Check if an observation operator is provided.
        if h_operator is None:

            # In the default case we simply set it to '1'.
            self.h_operator = np.array(1.0)
        else:

            # Here we copy the input array.
            self.h_operator = np.asarray(h_operator, dtype=float)
        # _end_if

        # Infer the system dimensions from the
        # diffusion noise covariance array.
        if self.sigma.ndim in (0, 1):

            # The process is scalar (i.e. D=1).
            self.dim_D = 1

        else:

            # The matrix should be square.
            self.dim_D = self.sigma.shape[0]

        # _end_if_

        # Infer the observations dimensions.
        if self.obs.ndim in (0, 1):

            # In this case the observations
            # are scalars (i.e. d=1).
            self.dim_d = 1

        else:

            # The dimensions of the observations vector.
            self.dim_d = self.obs.shape[1]

        # _end_if_

        # The first dimension (index=0) should always be
        # the observation (discrete) time-dimension.
        self.num_m = self.obs.shape[0]

        # Sanity check.
        if self.dim_d > self.dim_D:
            raise RuntimeError(f" {self.__class__.__name__}: System dimensions."
                               f" {self.dim_d} should be <= to {self.dim_D}.")
        # _end_if_

        # Number of mean points (for a 3rd order polynomial).
        self.num_mp = self.dim_D * (3 * self.num_m + 4)

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

        # Update the inverse of the coefficient.
        self.sigma_inverse, _ = cholesky_inv(self.sigma)
    # _end_def_

    def E_kl0(self, m0: array_t, s0: array_t):

        # Inverse of tau0 matrix.
        inv_tau0, _ = cholesky_inv(self._tau0)

        # Inverted.
        inv_s0, _ = cholesky_inv(s0)

        # Energy of the initial moment.
        z0 = m0 - self._mu0

        # Energy of the initial moment.
        E0 = 0.5 * (log_det(self._tau0.dot(inv_s0)) +
                    np.sum(np.diag(inv_tau0.dot(z0.T.dot(z0) +
                                                s0 - self._tau0))))
        # Kullback-Liebler value at t=0.
        return E0

    # _end_def_

    def E_sde(self, mean_pts, var_pts):

        # Number of discrete intervals (between observations).
        L = self.tx.size - 1

        # Preallocate for efficiency.
        w = np.zeros(L)

        # Calculate energy from all 'L' time intervals.
        for n in range(L):

            # Take the limits of the time interval.
            ti, tj = self.tx[n], self.tx[n+1]

            # Distance between the two limits.
            delta_t = np.abs(tj-ti)

            # Mid-point intervals (for the evaluation of the Esde function).
            h = float(delta_t/3.0)
            c = float(delta_t/2.0)

            # Separate variables for efficiency.
            vMn = mean_pts[:, ((n*3)-2):((n*3)+1)]
            vSn = var_pts[:, ((n*2)-1):((n*2)+1)]

            # Precompute for efficiency.
            energy_n = self.drift_fun(h, c, vMn, vSn, self.theta, self.sigma)

            # Scale with the inverse system noise.
            w[n] = energy_n.dot(self.sigma_inverse)
        # _end_for_

        # Return the total energy (from all intervals).
        return 0.5*np.sum(w)
    # _end_def_

    def E_obs(self, mean_pts, var_pts):

        Tkm, Tks = [], []

        # Logarithm of 2*pi.
        log2pi = 1.8378770664093453

        # Inverted Ri and Cholesky factor Qi.
        Ri, Qi = cholesky_inv(self.noise)

        # Keep the diagonal elements of Covariance.
        diagonal_Ri = Ri.diagonal()

        # Auxiliary quantity.
        Z = Qi.dot(self.obs - self.h_operator.dot(mean_pts[Tkm]))

        # Initial observations' energy.
        Eobs = 0

        # Calculate from all 'M' observations.
        for n in range(self.num_m):
            # Take the index of the observation.
            tk = Tks[n]

            # Get the auxiliary value at this point.
            Zn = Z[n]

            # Compute the energy of the tk-th observation.
            Eobs += Zn.T.dot(Zn) + diagonal_Ri.T.dot(self.h_operator.dot(var_pts[tk]))
        # _end_for_

        return 0.5*(Eobs + self.num_m*(self.dim_d*log2pi + log_det(self.noise)))
    # _end_def_

    def E_total(self, x):

        # Separate the mean from the variance points.
        mean_points = np.reshape(x[0:self.num_mp],
                                 self.dim_D, (3*self.num_m + 4))

        # The variance points are in log-space to ensure positivity.
        var_points = np.reshape(np.exp(x[self.num_mp + 1:]),
                                self.dim_D, (2*self.num_m + 3))

        # Return the total (free) energy as the sum of the individual
        # components.
        return self.E_kl0(mean_points[0], var_points[0]) +\
               self.E_sde(mean_points, var_points) +\
               self.E_obs(mean_points, var_points)
    # _end_def_

    def __call__(self, *args, **kwargs):
        return self.E_total(*args, **kwargs)
    # _end_def_

# _end_class_