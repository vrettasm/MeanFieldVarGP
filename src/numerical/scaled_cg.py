import numpy as np
from numba import njit
from time import perf_counter
from numpy import array as array_t


class SCG(object):
    """
    This class creates a Scaled Conjugate Gradient (SCG) optimization object.
    The goal is to find a local minimum of the function f(x). Here the input
    to 'f' (i.e. 'x0'), is a column vector.

    The minimisation process uses also the gradient df(x)/dx. To speed up the
    process, the input function return this gradient, along with the function
    value (thus we have less function calls).
    """

    __slots__ = ("f", "nit", "x_tol", "f_tol", "display", "stats")

    def __init__(self, func: callable, *args):
        """
        Default constructor the SCG class. It sets up the parameters
        for the optimization, such as: 1) number of max iterations,
        2) thresholds, etc.

        :param func: is the objective function to be optimised.

        :param args: is a dictionary containing all additional
                     parameters for 'func'.
        """

        # Check if we have given parameters.
        p_list = args[0] if args else {}

        # The function must be callable object.
        if callable(func):

            # Copy the function.
            self.f = func

        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Function {func} must be callable.")
        # _end_if_

        # Maximum number of iterations.
        self.nit = p_list["max_it"] if "max_it" in p_list else 500

        # Error tolerance in 'x'.
        self.x_tol = p_list["x_tol"] if "x_tol" in p_list else 1.0e-6

        # Error tolerance in 'fx'.
        self.f_tol = p_list["f_tol"] if "f_tol" in p_list else 1.0e-8

        # Display statistics flag.
        self.display = p_list["display"] if "display" in p_list else False

        # Statistics dictionary.
        self.stats = None
    # _end_def_

    @property
    def maxit(self):
        """
        Maximum number of iterations.

        :return: the 'nit' parameter.
        """
        return self.nit

    # _end_def_

    @property
    def xtol(self):
        """
        Tolerance in 'x'.

        :return: the 'x_tol' parameter.
        """
        return self.x_tol

    # _end_def_

    @property
    def ftol(self):
        """
        Tolerance in 'f(x)'.

        :return: the 'f_tol' parameter.
        """
        return self.f_tol

    # _end_def_

    @property
    def statistics(self):
        """
        Accessor method.

        :return: the statistics dictionary.
        """

        # Sanity check.
        if self.stats is None:
            raise NotImplementedError(f" {self.__class__.__name__}:"
                                      f" Stats dictionary has not been created.")
        # _end_if_

        return self.stats

    # _end_def_

    def __call__(self, x0: array_t, *args):
        """
        The call of the object itself will start the optimization.

        :param x0: Initial search point.

        :param args: additional function / gradient parameters.

        :return: 1)  x: the point where the minimum was found
                 2) fx: the function value (at the minimum point)
        """

        # Reset the stats in the object.
        self.stats = None

        # Local dictionary with statistical information.
        _stats = {"MaxIt": self.nit, "fx": np.zeros(self.nit, dtype=float),
                  "dfx": np.zeros(self.nit, dtype=float), "func_eval": 0}

        @njit(fastmath=True)
        def _fast_sum_abs(x_in: array_t):
            """
            Nested numba version of numpy functions.

            :param x_in: input array (dim,)

            :return: a much faster version of sum(abs(.)).
            """
            return np.sum(np.abs(x_in))
        # _end_def_

        # Check for verbosity.
        if self.display:
            print("SCG: Optimization started ...")
        # _end_if_

        # Localize 'f'.
        func = self.f

        # Localize function.
        _copy_to = np.copyto

        # Make sure input is flat.
        x = x0.flatten()

        # Size of input array.
        dim_x = x.size

        # Initial sigma.
        sigma0 = 1.0e-3

        # Initial function/gradients value.
        f_now, grad_new = func(x, *args)

        # Initialize grad_old vector.
        grad_old = np.copy(grad_new)

        # Increase function evaluations by one.
        _stats["func_eval"] += 1

        # Store the current values (fx / dfx).
        f_old = f_now

        # Set the initial search direction.
        d = -grad_new

        # Forces the calculation of directional derivatives.
        success = True

        # Counts the number of successes.
        count_success = 0

        # Initial scale parameter.
        beta = 1.0

        # Lower & Upper bounds on scale (beta).
        beta_min, beta_max = 1.0e-15, 1.0e+100

        # Initialization of parameters.
        kappa, theta, mu = 0.0, 0.0, 0.0

        # Get the machine precision constant.
        eps_float = np.finfo(float).eps

        # Start the timer.
        time_t0 = perf_counter()

        # Main optimization loop.
        for j in range(self.nit):

            # Calculate 1-st and 2-nd
            # directional derivatives.
            if success:
                # Inner-product.
                mu = d.T.dot(grad_new)

                if mu >= 0.0:
                    d = -grad_new
                    mu = d.T.dot(grad_new)
                # _end_if_

                # Compute kappa.
                kappa = d.T.dot(d)

                # Check for termination.
                if kappa < eps_float:
                    # Copy the value.
                    fx = f_now

                    # Update the statistic.
                    _stats["MaxIt"] = j+1

                    # Update object stats.
                    self.stats = _stats

                    # Exit from here.
                    return x, fx
                # _end_if_

                # Update sigma and check the gradient on a new direction.
                sigma = sigma0 / np.sqrt(kappa)
                x_plus = x + (sigma * d)

                # We evaluate the df(x_plus).
                _, g_plus = func(x_plus)

                # Increase function evaluations.
                _stats["func_eval"] += 1

                # Compute theta.
                theta = (d.T.dot(g_plus - grad_new)) / sigma
            # _end_if_

            # Increase effective curvature and evaluate step size alpha.
            delta = theta + (beta * kappa)
            if delta <= 0.0:
                delta = beta * kappa
                beta = beta - (theta / kappa)
            # _end_if_

            # Update 'alpha'.
            alpha = -(mu / delta)

            # Evaluate the function at a new point.
            x_new = x + (alpha * d)

            # Evaluate fx and dfx at the new point.
            # NOTE: Because we haven't accepted yet this position as the
            # next 'x' state, we use the 'g_now' to store the gradient.
            f_new, g_now = func(x_new)

            # Note that the gradient is computed anyway.
            _stats["func_eval"] += 1

            # Calculate the new comparison ratio.
            Delta = 2.0 * (f_new - f_old) / (alpha * mu)
            if Delta >= 0.0:

                # Set the flag.
                success = True

                # Update counter.
                count_success += 1

                # Copy the new values.
                f_now = f_new

                # Update the new search position.
                _copy_to(x, x_new)

            else:

                # Cancel the flag.
                success = False

                # Copy the old values.
                f_now = f_old

                # Update the gradient vector.
                _copy_to(g_now, grad_old)

            # _end_if_

            # Total gradient: j-th iteration.
            total_grad = _fast_sum_abs(g_now)

            # Store statistics.
            _stats["fx"][j] = f_now
            _stats["dfx"][j] = total_grad

            # Used in verbose/display mode.
            if self.display and (np.mod(j, 50) == 0):

                # Timer snapshot after 'j' iterations.
                time_tj = perf_counter()

                # Print the current info.
                print(f"It= {j:>5}: F(x)= {f_now:.3E} -/- "
                      f"Sum(Gradients)= {total_grad:.3E} -/- "
                      f"Delta(Elapsed)= {time_tj-time_t0:.2f} sec.")

                # Assign the current time to 't0'.
                time_t0 = time_tj

            # _end_if_

            # Check for success.
            if success:

                # Check for termination.
                if (np.abs(alpha * d).max() <= self.x_tol) and\
                        (np.abs(f_new - f_old) <= self.f_tol):
                    # Copy the new value.
                    fx = f_new

                    # Update the statistic.
                    _stats["MaxIt"] = j + 1

                    # Update object stats.
                    self.stats = _stats

                    # Exit.
                    return x, fx
                else:
                    # Update variables for the new position.
                    f_old = f_new

                    # Copy the "new" gradient to the "old".
                    _copy_to(grad_old, grad_new)

                    # Evaluate function/gradient at the new point.
                    f_now, grad_new = func(x, *args)

                    # Increase function evaluations by one.
                    _stats["func_eval"] += 1

                    # If the gradient is zero then exit.
                    if np.isclose(grad_new.T.dot(grad_new), 0.0):
                        # Copy the new value.
                        fx = f_now

                        # Update the statistic.
                        _stats["MaxIt"] = j + 1

                        # Update object stats.
                        self.stats = _stats

                        # Exit.
                        return x, fx
                # _end_if_

            # _end_if_

            # Adjust beta according to comparison ratio.
            if Delta < 0.25:
                beta = np.minimum(4.0 * beta, beta_max)
            # _end_if_

            if Delta > 0.75:
                beta = np.maximum(0.5 * beta, beta_min)
            # _end_if_

            # Update search direction using Polak-Ribiere formula
            # or re-start in direction of negative gradient after
            # 'dim_x' steps.
            if count_success == dim_x:
                d = -grad_new
                count_success = 0
            else:
                # Check the flag.
                if success:
                    gamma = np.maximum(grad_new.T.dot(grad_old - grad_new) / mu, 0.0)
                    d = (gamma * d) - grad_new
                # _end_if_

            # _end_if_

        # _end_for_

        # Display a final (warning) to the user.
        print(f"SGC: Maximum number of iterations ({self.nit}) has been reached.")

        # Here we have reached the maximum number of iterations.
        fx = f_old

        # Update object stats.
        self.stats = _stats

        # Exit from here.
        return x, fx
    # _end_def_

    def __str__(self):
        """
        Override to print a readable string presentation
        of the object. This will include its id(), along
        with its fields values.

        :return: a string representation of a SCG object.
        """

        return f"SCG Id({id(self)}): "\
               f"Function={self.f}, Max-It={self.nit}, " \
               f"x_tol={self.x_tol}, f_tol={self.f_tol}"
    # _end_def_

# _end_class_
