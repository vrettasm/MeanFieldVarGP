import time
import numpy as np
from numpy import zeros
from numpy import array as array_t
from numpy import reshape as reshape
from numpy import squeeze as squeeze
from joblib import Parallel, delayed
from scipy.integrate import quad_vec
from scipy.optimize import check_grad

from numerical.scaled_cg import SCG
from numerical.utilities import cholesky_inv, log_det
from dynamical_systems.stochastic_process import StochasticProcess


class FreeEnergy(object):

    # Declare all the class variables here.
    __slots__ = ("drift_fun_sde", "grad_fun_mp", "grad_fun_vp", "_mu0", "_tau0",
                 "theta", "sigma", "tk", "obs_times", "obs_noise", "obs_values",
                 "obs_mask", "dim_D", "dim_d", "num_M", "num_mp", "ikm", "iks",
                 "ix_ikm", "ix_iks")

    def __init__(self, sde: StochasticProcess, mu0: array_t, tau0: array_t,
                 obs_times: array_t, obs_values: array_t, obs_noise: array_t,
                 obs_mask: list = None):
        """
        Default constructor of the FreeEnergy class.

        :param sde: StochasticProcess (SDE).

        :param mu0: (prior) mean value(s) at time t=0.

        :param tau0: (prior) variance value(s) at time t=0.

        :param obs_times: (discrete) time-window of observation times.

        :param obs_values: observation values (including noise).

        :param obs_noise: observation noise variance.

        :param obs_mask: observation mask (default = None).
        This is a list that defines which dimensions are observed and
        which are not. By default, all states are assumed to be observed.
        """

        # (WRAPPER FUNCTION): SDE energy function.
        self.drift_fun_sde = sde.energy

        # (WRAPPER FUNCTION): Gradients of the
        # SDE with respect to the mean points.
        self.grad_fun_mp = sde.grad_mean

        # (WRAPPER FUNCTION): Gradients of the
        # SDE with respect to the variance points.
        self.grad_fun_vp = sde.grad_variance

        # Prior mean (t=0).
        self._mu0 = np.asarray(mu0, dtype=float)

        # Prior variance diagonal elements (t=0).
        self._tau0 = np.asarray(tau0, dtype=float)

        # Make sure the SDE drift and diffusion noise
        # parameters are entered correctly as arrays.
        self.theta = sde.theta
        self.sigma = sde.sigma

        # Discrete time-window: Tk=[t0, tf].
        self.tk = sde.time_window

        # Sanity check.
        if not np.all(np.diff(self.tk) > 0.0):
            raise ValueError(f" {self.__class__.__name__}:"
                             f" Time window [t0, tf] is not increasing.")
        # _end_if_

        # Make sure the observation related parameters are
        # entered correctly as numpy arrays (type = float).
        self.obs_times = np.asarray(obs_times, dtype=float)
        self.obs_noise = np.asarray(obs_noise, dtype=float)
        self.obs_values = np.asarray(obs_values, dtype=float)

        # Infer the system dimensions.
        self.dim_D = self.sigma.size

        # Check if an observation operator is provided.
        if obs_mask is None:

            # Make a list with dim_D "True" values.
            self.obs_mask = self.dim_D * [True]
        else:

            # Sanity check.
            if len(obs_mask) != self.dim_D:
                raise RuntimeError(f" {self.__class__.__name__}:"
                                   f" Observation mask mismatch {len(obs_mask)} != {self.dim_D}")
            # _end_if_

            # Sanity check.
            if all(isinstance(item, bool) for item in obs_mask):

                # Here we copy the list.
                self.obs_mask = obs_mask

            else:
                raise ValueError(f" {self.__class__.__name__}:"
                                 f" Observation mask contains non-boolean values!")
            # _end_if_

            # Sanity check.
            if sum(self.obs_mask) == 0:
                raise RuntimeError(f" {self.__class__.__name__}:"
                                   f" Observation mask contains only False values.")
            # _end_if_

        # _end_if_

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
                               f" {self.dim_d} should be less than, or equal,"
                               f" to {self.dim_D}.")
        # _end_if_

        # Sanity check.
        if self.num_M != self.obs_values.shape[0]:
            raise RuntimeError(f" {self.__class__.__name__}:  System dimensions."
                               f" Observation times {self.num_M} should be equal"
                               f" to observation values {self.obs_values.shape[0]}.")
        # _end_if_

        # Total number of mean points (for a 3rd order polynomial)
        # scaled with the dimensions.
        self.num_mp = self.dim_D * (3 * self.num_M + 4)

        # Indexes of the observations at the mean/variance points.
        self.ikm = [3*i for i in range(1, self.num_M+1)]
        self.iks = [2*i for i in range(1, self.num_M+1)]

        # This mesh-grid of indexes is used in E_cost
        # method to map the observation gradients.
        self.ix_ikm = np.ix_(self.obs_mask, self.ikm)
        self.ix_iks = np.ix_(self.obs_mask, self.iks)

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

        NOTE: This should be used only if we optimize
        the prior moments.

        :param new_value: the new value we want to set.

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

        NOTE: This should be used only if we optimize
        the prior moments.

        :param new_value: the new value we want to set.

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

        NOTE: This should be used only if we optimize
        the drift (theta) parameters.

        :param new_value: the new value we want to set.

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

        NOTE: This should be used only if we optimize
        the diffusion (Sigma) parameters.

        :param new_value: the new value we want to set.

        :return: None.
        """
        self.sigma = np.asarray(new_value, dtype=float)
    # _end_def_

    def E_kl0(self, m0: array_t, s0: array_t):
        """
        Energy of the initial state with a Gaussian
        prior q(x|t=0).

        :param m0: marginal mean at t=0, (dim_D).

        :param s0: marginal variance at t=0, (dim_D).

        :return: energy of the initial state E0, (scalar)
        and its derivatives with respect to 'm0' and 's0'.
        """

        # Difference of the two "mean" vectors.
        z0 = m0 - self._mu0

        # Energy of the initial moment.
        # NOTE: This formula works only because the matrices 'tau0'
        # and 's0' are diagonal, and we work with vectors.
        E0 = log_det(self._tau0/s0) + np.sum((z0**2 + s0 - self._tau0) / self._tau0)

        # Sanity check.
        if not np.isfinite(E0):
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" E0 is not a finite number: {E0}")
        # _end_if_

        # Auxiliary variable.
        one_ = np.ones(1)

        # Compute the gradients. We use the "atleast_1d" to ensure
        # that the scalar cases (even though rare) will be covered.
        dE0_dm0 = np.atleast_1d(z0 / self._tau0)
        dE0_ds0 = np.atleast_1d(0.5 * (one_ / self._tau0 - one_ / s0))

        # Kullback-Liebler and its derivatives at
        # time t=0, (i.e. dE0/dm(0), dE0/ds(0))
        return 0.5 * E0, dE0_dm0, dE0_ds0

    # _end_def_

    @staticmethod
    def single_interval(n_th: int, ti: float, tj: float, drift_func: callable, gradMP_func: callable,
                        gradSP_func: callable, mean_pts: array_t, vars_pts: array_t, sigma: array_t,
                        theta: array_t, inv_sigma: array_t):
        """
        This static method computes the integrated values of the Esde and its gradients for a single
        time interval [ti, tj].

        :param n_th: this integer value represent the n-th interval of integration. It is used mainly
        to guarantee the order in which we return the values from the Parallel loop.

        :param ti: this is the first limit of integration in the [ti, tj] interval.

        :param tj: this is the second limit of integration in the [ti, tj] interval.

        :param drift_func: this is the Esde (drift) function.

        :param gradMP_func: this is the dEsde_dm function.

        :param gradSP_func: this is the dEsde_ds function.

        :param mean_pts: mean (polynomial) points in the n-th interval.

        :param vars_pts: variance (polynomial) points in the n-th interval.

        :param sigma: diffusion noise parameters vector.

        :param theta: drift model parameters vector.

        :param inv_sigma: inverse values of diffusion noise vector.

        :return: the integrated values (in [ti, tj] of Esde, dEsde_dm and dEsde_ds.
        """

        # NOTE: This should not change for equally spaced
        # observations. This is here to ensure that small
        # 'dt' deviations will not affect the algorithm.
        delta_t = np.abs(tj-ti)

        # Mid-point intervals (for the evaluation of the Esde function).
        # NOTE: These should not change for equally spaced observations.
        h = float(delta_t/3.0)
        c = float(delta_t/2.0)

        # Pack the total set of input parameters.
        # Note: If everything is done right then:
        #   1) tj == ti + (3 * h)
        #   2) tj == ti + (2 * c)
        params = [ti, ti + h, ti + (2 * h), tj,
                  ti, ti + c, tj,
                  *mean_pts.ravel(order='C'),
                  *vars_pts.ravel(order='C'),
                  *sigma, *theta]

        # We use the lambda functions here to fix all the
        # additional input parameters except the time "t".
        Esde = quad_vec(lambda t: drift_func(t, *params), ti, tj,
                        limit=100, epsabs=1.0e-06, epsrel=1.0e-06)[0]

        # Solve the integrals of dEsde(t)/dMp in [ti, tj].
        integral_dEn_dm = quad_vec(lambda t: gradMP_func(t, *params), ti, tj,
                                   limit=100, epsabs=1.0e-06, epsrel=1.0e-06)[0]

        # Solve the integrals of dEsde(t)/dSp in [ti, tj].
        integral_dEn_ds = quad_vec(lambda t: gradSP_func(t, *params), ti, tj,
                                   limit=100, epsabs=1.0e-06, epsrel=1.0e-06)[0]
        # Sanity check.
        if inv_sigma.size == 1:

            # Remove singleton dimension.
            Esde = squeeze(inv_sigma*Esde)

            # This way we avoid errors in 1D systems.
            dEsde_dm = inv_sigma*integral_dEn_dm
            dEsde_ds = inv_sigma*integral_dEn_ds

        else:

            # Scale everything with inverse noise.
            Esde = inv_sigma.dot(Esde)

            # NOTE: the correct dimensions are (D x 4).
            dEsde_dm = inv_sigma.dot(integral_dEn_dm)

            # NOTE: the correct dimensions are (D x 3).
            dEsde_ds = inv_sigma.dot(integral_dEn_ds)

        # _end_if_

        # Include the n-th interval here to guarantee the order
        # of the sub-intervals.
        return n_th, 0.5 * Esde, 0.5 * dEsde_dm, 0.5 * dEsde_ds
    # _end_def_

    def E_sde(self, mean_pts, vars_pts):
        """
        Energy from the SDE prior process.

        :param mean_pts: optimized mean points.

        :param vars_pts: optimized variance points.

        :return: Energy from the SDE prior process (scalar) and
        its gradients with respect to the mean & variance points.
        """

        # Local copy of the single interval function.
        _single_interval = self.single_interval

        # Get the system dimensions once.
        dim_D = self.dim_D

        # Local observation times.
        obs_times = self.obs_times

        # Number of discrete intervals
        # (between the observations).
        L = obs_times.size - 1

        # Initialize energy for the SDE.
        Esde = 0.0

        # Initialize the gradients arrays.
        # -> dEsde_dm := dEsde(tk)/dm(tk)
        # -> dEsde_ds := dEsde(tk)/ds(tk)
        dEsde_dm = zeros((L, 4*dim_D), dtype=float)
        dEsde_ds = zeros((L, 3*dim_D), dtype=float)

        # Inverted diagonal noise vector.
        inv_sigma = np.atleast_1d(1.0 / self.sigma)

        # NOTE: For small systems (e.g. D < 8) it is
        # preferred to use as backend "threading".
        back_end = "loky" if dim_D > 8 else "threading"

        # Run the 'L' intervals in parallel.
        results = Parallel(n_jobs=4, backend=back_end)(

            delayed(_single_interval)(n, obs_times[n], obs_times[n+1],
                                      self.drift_fun_sde, self.grad_fun_mp, self.grad_fun_vp,
                                      mean_pts[:, (3 * n): (3 * n) + 4],
                                      vars_pts[:, (2 * n): (2 * n) + 3],
                                      self.sigma, self.theta, inv_sigma) for n in range(L)
        )

        # Extract all the result from the parallel run.
        # NOTE: The order of the results matters in the
        # computation of the gradients!
        for res_ in results:
        
            # This is the sub-interval.
            i = res_[0]

            # Accumulate the Esde here.
            Esde += res_[1]

            # The gradients will be accumulated in the E_cost.
            dEsde_dm[i] = res_[2]
            dEsde_ds[i] = res_[3]
        # _end_for_

        # Sanity check.
        if not np.isfinite(Esde):
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Esde is not a finite number: {Esde}")
        # _end_if_

        # Return the total energy (including the correct scaling).
        # and its gradients with respect ot the mean and variance
        # (optimized) points.
        return Esde,\
               reshape(dEsde_dm, (L, dim_D, 4), order='C'),\
               reshape(dEsde_ds, (L, dim_D, 3), order='C')
    # _end_def_

    def E_obs(self, mean_pts, vars_pts):
        """
        Energy from the Gaussian likelihood.

        :param mean_pts: optimized mean points.

        :param vars_pts: optimized variance points.

        :return: Energy from the observation likelihood (scalar)
        and its gradients with respect to the mean and variance
        points.
        """

        # Inverted Ri and Cholesky factor Qi.
        Ri, Qi = cholesky_inv(self.obs_noise)

        # Sanity check.
        if isinstance(Qi, float):
            Qi = np.array(Qi)
        # _end_if_

        # Sanity check.
        if isinstance(Ri, float):
            Ri = np.atleast_2d(Ri)
        # _end_if_

        # Auxiliary quantity: (Y - H*m).
        Y_minus_Hm = self.obs_values.T - mean_pts

        # Auxiliary quantity (for the E_obs).
        Z = Qi.dot(Y_minus_Hm)

        # Auxiliary quantity (for the E_obs).
        W = Ri.diagonal().dot(vars_pts)

        # These are the derivatives of E_{obs} w.r.t. the mean points.
        kappa_1 = -Ri.dot(Y_minus_Hm).T

        # Note that the dEobs(k)/ds(k) is identical for all observations.
        kappa_2 = 0.5 * Ri.diagonal()

        # Initialize observations' energy.
        Eobs = 0.0

        # Initialize gradients arrays.
        dEobs_dm = zeros((self.dim_d, self.num_M), dtype=float)
        dEobs_ds = zeros((self.dim_d, self.num_M), dtype=float)

        # Remove singleton dimensions.
        if self.dim_d == 1:
            kappa_1 = squeeze(kappa_1)
        # _end_if_

        # Calculate partial energies from all 'M' observations.
        # NOTE: The gradients are given by:
        #   1. dEobs(k)/dm(k) := -H'*Ri*(yk-h(xk))
        #   2. dEobs(k)/ds(k) := 0.5*diag(H'*Ri*H)
        for k in range(self.num_M):

            # Get the auxiliary value.
            Zk = Z[:, k]

            # Compute the energy of the k-th observation.
            Eobs += Zk.T.dot(Zk) + W[k]

            # Gradient of E_{obs} w.r.t. m(tk).
            dEobs_dm[:, k] = kappa_1[k]

            # Gradient of E_{obs} w.r.t. S(tk).
            dEobs_ds[:, k] = kappa_2
        # _end_for_

        # Logarithm of 2*pi.
        log2pi = 1.8378770664093453

        # Final energy value (including the constants).
        Eobs += self.num_M * (self.dim_d * log2pi + log_det(self.obs_noise))

        # Sanity check.
        if not np.isfinite(Eobs):
            raise RuntimeError(f" {self.__class__.__name__}:"
                               f" Eobs is not a finite number: {Eobs}")
        # _end_if_

        # Remove singleton dimensions.
        if self.dim_d == 1:
            dEobs_dm = squeeze(dEobs_dm)
            dEobs_ds = squeeze(dEobs_ds)
        # _end_if_

        # Return the total observation energy
        # and its gradients.
        return 0.5 * Eobs, dEobs_dm, dEobs_ds
    # _end_def_

    def E_cost(self, x, output_gradients=True):
        """
        Total cost function value (scalar) and derivatives.
        This is passed to the "scaled_cg" optimization:

         --> scaled_cg(E_cost, x0)

        other alternatives are: BFGS, CG, Newton-CG, L-BFGS-B
        from the scipy.optimize module.

        :param x: the optimization variables. Here we use the
        mean and variance points of the Lagrange polynomials.

        :param output_gradients: boolean flag to whether include,
        or not the gradients in the output. This is used when the
        optimize method requires as output only the function f(x)
        or also the gradient df(x)/dx.

        NOTE: the gradients are computed always regardless of the
        boolean "output_gradients". This is because it speeds up
        the minimization function (less function calls).

        :return: total energy value and derivatives (with respect
        to the input mean and variance points).
        """

        # Separate the mean from the variance points.
        mean_points = reshape(x[0:self.num_mp],
                              (self.dim_D, (3*self.num_M + 4)),
                              order='C')

        # The variance points are in log-space to ensure positivity,
        # so we pass them through the exponential function first.
        vars_points = reshape(np.exp(x[self.num_mp:]),
                              (self.dim_D, (2*self.num_M + 3)),
                              order='C')

        # Energy (and gradients) from the initial moment (t=0).
        E0, dE0_dm0, dE0_ds0 = self.E_kl0(mean_points[:, 0],
                                          vars_points[:, 0])

        # Energy from the SDE (and gradients).
        Esde, dEsde_dm, dEsde_ds = self.E_sde(mean_points,
                                              vars_points)

        # Energy from the observations' likelihood (and gradients).
        Eobs, dEobs_dm, dEobs_ds = self.E_obs(mean_points[self.ix_ikm],
                                              vars_points[self.ix_iks])
        # Put all energy values together.
        Ecost = E0 + Esde + Eobs

        # Check if we want the gradients to be returned.
        if not output_gradients:

            # Exit here.
            return Ecost
        # _end_if_

        # Put all gradients together.
        Ecost_dm = zeros((self.dim_D, 3 * self.num_M + 4), dtype=float)
        Ecost_ds = zeros((self.dim_D, 2 * self.num_M + 3), dtype=float)

        # Copy the gradients of the first interval.
        Ecost_dm[:, 0:4] = dEsde_dm[0]
        Ecost_ds[:, 0:3] = dEsde_ds[0]

        # Iterate over the rest (L-1) intervals.
        for n in range(1, self.obs_times.size - 1):

            # Add the link between intervals, at
            # observation times.
            Ecost_dm[:, (3 * n)] += dEsde_dm[n][:, 0]
            Ecost_ds[:, (2 * n)] += dEsde_ds[n][:, 0]

            # Copy the rest of the gradients.
            Ecost_dm[:, (3 * n+1): (3 * n) + 4] = dEsde_dm[n][:, 1:]
            Ecost_ds[:, (2 * n+1): (2 * n) + 3] = dEsde_ds[n][:, 1:]

        # _end_for_
        
        # Add the initial contribution from E0.
        Ecost_dm[:, 0] += dE0_dm0
        Ecost_ds[:, 0] += dE0_ds0

        # Add the gradients (at observation times).
        Ecost_dm[self.ix_ikm] += dEobs_dm
        Ecost_ds[self.ix_iks] += dEobs_ds

        # Rescale the variance gradients to account for
        # the log-transformation and ensure positivity.
        # NOTE: This is element-wise multiplication !!!
        Ecost_ds *= vars_points

        # Return the total (free) energy as the sum of the individual
        # components. NOTE: If we want to optimize the hyperparameter
        # we should add another term, e.g. E_param, and include it in
        # the total sum of energy values.
        return Ecost, np.concatenate((Ecost_dm.ravel(order='C'),
                                      Ecost_ds.ravel(order='C')), axis=0)
    # _end_def_

    def find_minimum(self, x0, maxiter: int = 100, x_tol: float = 1.0e-5,
                     f_tol: float = 1.0e-5, check_gradients=False, verbose=False):
        """
        Optimizes the free energy value (E_cost) by using the Scaled
        Conjugate Gradient (SGC) optimizer. The result is the final
        set of optimization variables.

        :param x0: initial set of variables to start the minimization.

        :param maxiter: (int) number of iterations in the minimization.

        :param x_tol: (float) tolerance between two successive solutions
        x_{k} and x_{k+1}.

        :param f_tol: (float) tolerance between two successive function
        evaluations f(x_{k}) and f(x_{k+1}).

        :param check_gradients: (boolean) flag to determine the checking
        of the gradients, before and after the minimization.

        :param verbose: (boolean) flag to display information about the
        convergence of the process.

        :return: the optimal solution found by SGC().
        """

        def _analytic_grad_func(xin: array_t):
            """
            Locally defined gradient function.

            Used only if check_gradients=True.

            :param xin: The input we want to get
                        the analytic gradient at.
            """

            # Runs the E_cost() with the default
            # setting (i.e. output_gradients=True).
            _, grad_A = self.E_cost(xin)

            # Get the "analytic" gradient.
            return grad_A

        # _end_def_

        # Check numerically the gradients.
        if check_gradients:

            # Display the action.
            print("Grad-Check |BEFORE| minimization ...")

            # Get the grad-check error.
            error_t0 = check_grad(lambda x_in: self.E_cost(x_in, output_gradients=False),
                                  _analytic_grad_func, x0.copy())

            # Display the error.
            print(f" > Error = {error_t0:.3E}")
            print("------------------------------------\n")
        # _end_if_

        # Ensure optimization variables have the correct types.
        # Put lower limits to them to avoid invalid entries.
        maxiter = np.maximum(int(maxiter), 10)
        x_tol = np.maximum(float(x_tol), 1.0e-20)
        f_tol = np.maximum(float(f_tol), 1.0e-20)

        # Setup SCG options.
        options = {"max_it": maxiter, "x_tol": x_tol, "f_tol": f_tol,
                   "display": verbose}

        # Create an SCG optimizer.
        scg_minimize = SCG(self.E_cost, options)

        # Start the timer.
        time_t0 = time.perf_counter()

        # Run the optimization procedure.
        opt_x, opt_fx = scg_minimize(x0)

        # Stop the timer.
        time_tf = time.perf_counter()

        # Print final duration in seconds.
        print(f"Elapsed time: {(time_tf - time_t0):.2f} seconds.\n")

        # Check numerically the gradients.
        if check_gradients:

            # Display the action.
            print("Grad-Check |AFTER| minimization ...")

            # Get the grad-check error.
            error_tf = check_grad(lambda x_in: self.E_cost(x_in, output_gradients=False),
                                  _analytic_grad_func, opt_x.copy())

            # Display the error.
            print(f" > Error = {error_tf:.3E}")
            print("------------------------------------\n")

        # _end_if_

        # Final message.
        print("Done!")

        # Get the final (optimal) results.
        # We also return a copy of the SCG
        # statistics for further analysis.
        return opt_x, opt_fx, scg_minimize.stats

    # _end_def_

# _end_class_
