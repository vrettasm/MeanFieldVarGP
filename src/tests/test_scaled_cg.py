import unittest
import numpy as np
from numpy.random import default_rng
from scipy.optimize import approx_fprime

# Import custom code.
from src.numerical.scaled_cg import SCG


class TestScaledCG(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> Testing Scaled-CG - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> Testing Scaled-CG - STOP -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with
        default random number generator.

        :return: None.
        """

        # Create the test with fixed seed.
        self.rng = default_rng(seed=795)

        # Fix the 'eps' here.
        self.eps = np.finfo(float).eps

    # _end_def_

    def test_sphere_func(self):
        """
        Test the Sphere Function (n >= 1).

            f(x) = \Sum_{i=1}^{n} x_i^2

        the global minimum is found at: f(0, 0, ..., 0) = 0.

        :return: None.

        """

        # Sphere function.
        f = lambda xk: np.sum(xk ** 2)

        # Gradient (numerical).
        df = lambda xk: approx_fprime(xk=xk, f=f,
                                      epsilon=self.eps)
        # Function to minimize.
        func = lambda x: (f(x), df(x))

        # Initial random points in (-oo, +oo).
        x0 = self.rng.standard_normal(5)

        # Create the SCG.
        optim_fun = SCG(func, {"max_it": 1000})

        # Run the optimization with default parameters.
        x_opt, fx_opt = optim_fun(x0)

        # The global minimum should be zero.
        self.assertTrue(np.allclose(fx_opt, 0.0),
                        msg="The global minimum should be zero.")

        # Also, the position 'x' should be zero.
        self.assertTrue(np.allclose(x_opt, 0.0, atol=1.0e-4),
                        msg="The minimum should be found at 0.")
    # _end_def_

    def test_rosenbrock_fun(self):
        """
        Test the Rosenbrock Function (n=2).

            f(x) = 100*(x1 - x0^2)^2 + (1 - x0)^2

        the global minimum is found at: f(1, 1) = 0.

        :return: None.
        """

        # Sphere function.
        f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        # Gradient (numerical).
        df = lambda xk: approx_fprime(xk=xk, f=f,
                                      epsilon=self.eps)
        # Function to minimize.
        func = lambda x: (f(x), df(x))

        # Initial random point.
        x0 = self.rng.random(2)

        # Create the SCG, with optional parameters.
        optim_fun = SCG(func, {"max_it": 1000})

        # Run the optimization with default parameters.
        x_opt, fx_opt = optim_fun(x0)

        # The global minimum should be zero.
        self.assertTrue(np.allclose(fx_opt, 0.0),
                        msg="The global minimum should be zero.")

        # The position 'x' should be one.
        self.assertTrue(np.abs(x_opt - 1.0).max() < 1.0E-3,
                        msg="The minimum should be found at 1.")
    # _end_def_


if __name__ == '__main__':
    unittest.main()
