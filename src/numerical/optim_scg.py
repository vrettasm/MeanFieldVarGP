import numpy as np


class SCG(object):
    """
    This class creates a Scaled Conjugate Gradient (SCG) optimization
    object. Attempts to find a local minimum of the function f(x).
    Here 'x0' is a column vector and 'f' must returns a scalar value.

    The minimisation process uses also the gradient df(x)/dx. To speed
    up the process, the input function is passing this gradient, along
    with the function value.

    The point at which 'f' has a local minimum is returned as 'x'. The
    function value at that point (the minimum) is returned in "fx".

    Additional information is passed to the 'stats' dictionary.
    """

    __slots__ = ("f", "nit", "x_tol", "f_tol", "display", "stats")

    def __init__(self, func, *args):
        """
        Default constructor the SCG class.

        :param func: is the objective function to be optimised.

        :param args: is a dictionary containing all additional
        parameters for 'func'.
        """

        # Check if we have given parameters.
        p_list = args[0] if args else {}

        # Function handle.
        self.f = func

        # Maximum number of iterations.
        if "max_it" in p_list:
            self.nit = p_list["max_it"]
        else:
            self.nit = 150
        # _end_if_

        # Error tolerance in 'x'.
        if "x_tol" in p_list:
            self.x_tol = p_list["x_tol"]
        else:
            self.x_tol = 1.0e-6
        # _end_if_

        # Error tolerance in 'fx'.
        if "f_tol" in p_list:
            self.f_tol = p_list["f_tol"]
        else:
            self.f_tol = 1.0e-8
        # _end_if_

        # Display statistics flag.
        if "display" in p_list:
            self.display = p_list["display"]
        else:
            self.display = False
        # _end_if_

        # Statistics dictionary.
        self.stats = {"MaxIt": self.nit, "fx": np.zeros(self.nit),
                      "dfx": np.zeros(self.nit), "f_eval": 0.0,
                      "df_eval": 0.0, "beta": np.zeros(self.nit)}
    # _end_def_

    def __call__(self, x0, *args):
        """
        The call of the object itself will enable the optimization.

        :param x0: Initial search point.

        :param args: additional function / gradient parameters.

        :return: 1)  x: the point where the minimum was found,
                 2) fx: the function value (at the minimum point).
        """
        # Make a local copy of the function.
        _copy = np.copy

        # Make sure input is flat.
        x = x0.flatten()

        # Size of input array.
        dim_x = x.size

        # Initial sigma.
        sigma0 = 1.0e-3

        # Initial function/gradients value.
        f_now, grad_new = self.f(x, *args)

        # Increase function / gradient evaluations by one.
        self.stats["f_eval"] += 1
        self.stats["df_eval"] += 1

        # Store the current values (fx / dfx).
        f_old, grad_old = f_now, _copy(grad_new)

        # Set the initial search direction.
        d = -grad_new

        # Force calculation of directional derivatives.
        success = 1

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

        # Main optimization loop.
        for j in range(self.nit):

            # Calculate 1st and 2nd
            # directional derivatives.
            if success == 1:
                # Inner-product.
                mu = d.T.dot(grad_new)

                if mu >= 0.0:
                    d = -grad_new
                    mu = d.T.dot(grad_new)
                # _end_if_

                # Compute kappa.
                kappa = d.T.dot(d)

                # And check for termination.
                if kappa < eps_float:
                    # Copy the value.
                    fx = f_now

                    # Update the statistic.
                    self.stats["MaxIt"] = j+1

                    # Exit from here.
                    return x, fx
                # _end_if_

                # Update sigma and check the gradient on a new direction.
                sigma = sigma0 / np.sqrt(kappa)
                x_plus = x + (sigma * d)

                # We evaluate the df(x_plus).
                _, g_plus = self.f(x_plus)

                # Increase function/gradients evaluations by one.
                self.stats["f_eval"] += 1
                self.stats["df_eval"] += 1

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

            # Return only the fx (not the dfx).
            f_new = self.f(x_new, False)

            # Note that the gradient is computed anyway.
            self.stats["f_eval"] += 1
            self.stats["df_eval"] += 1

            # Calculate the new comparison ratio.
            Delta = 2.0 * (f_new - f_old) / (alpha * mu)
            if Delta >= 0.0:
                success = 1
                count_success += 1
                x, f_now, g_now = _copy(x_new), _copy(f_new), _copy(grad_new)
            else:
                success = 0
                f_now, g_now = f_old, _copy(grad_old)
            # _end_if_

            # Total gradient.
            total_grad = np.sum(np.abs(g_now))

            # Store statistics.
            self.stats["fx"][j] = f_now
            self.stats["beta"][j] = beta
            self.stats["dfx"][j] = total_grad

            # Used in debugging mode.
            if self.display and (np.mod(j, 50) == 0):
                print(" {0}: fx={1:.3f}\tsum(gx)={2:.3f}".format(j, f_now, total_grad))
            # _end_if_

            # TBD:
            if success == 1:
                # Check for termination.
                if (np.abs(alpha * d).max() <= self.x_tol) and\
                        (np.abs(f_new - f_old) <= self.f_tol):
                    # Copy the new value.
                    fx = f_new

                    # Update the statistic.
                    self.stats["MaxIt"] = j + 1

                    # Exit.
                    return x, fx
                else:
                    # Update variables for the new position.
                    f_old, grad_old = f_new, _copy(grad_new)

                    # Evaluate function/gradient at the new point.
                    f_now, grad_new = self.f(x, *args)

                    # Increase function/gradients evaluations by one.
                    self.stats["f_eval"] += 1
                    self.stats["df_eval"] += 1

                    # If the gradient is zero then exit.
                    if np.isclose(grad_new.T.dot(grad_new), 0.0):
                        # Copy the new value.
                        fx = f_now

                        # Update the statistic.
                        self.stats["MaxIt"] = j + 1

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
                if success == 1:
                    gamma = np.maximum(grad_new.T.dot(grad_old - grad_new) / mu, 0.0)
                    d = (gamma * d) - grad_new
                # _end_if_
            # _end_if_
        # _end_for_

        # Display a final (warning) to the user.
        print(" SGC: Maximum number of iterations has been reached.")

        # Here we have reached the maximum number of iterations.
        fx = f_old

        # Exit from here.
        return x, fx
    # _end_def_

    @property
    def statistics(self):
        """
        Accessor method.

        :return: the statistics dictionary.
        """
        return self.stats
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation
        of the object. This will include its id(), along
        with its fields values.

        :return: a string representation of a SCG object.
        """

        return " SCG Id({0}): Function={1}, Max-It={2}," \
               " x_tol={3}, f_tol={4}".format(id(self), self.f, self.nit, self.x_tol, self.f_tol)
    # _end_def_

# _end_class_
