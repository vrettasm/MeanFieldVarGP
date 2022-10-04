import unittest
import numpy as np

# Import custom code
from src.dynamical_systems.lorenz_96 import Lorenz96


class TestLorenz96(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> Testing Lorenz96 - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> Testing Lorenz96 - STOP -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with fixed values.

        :return: None.
        """

        # System dimensions.
        self.D = 40

        # Drift parameter vector.
        self.theta = 8.0

        # Diffusion noise parameters vector.
        self.sigma = 40.0 * np.ones(self.D)

        # Create an object with fixed input values.
        self.test_obj = Lorenz96(sigma=self.sigma, theta=self.theta,
                                 dim_D=self.D, r_seed=911)
    # _end_def_

    def test_dimensions(self) -> None:
        """
        The number of system dimensions should
        be more than (or equal) to four.
        """

        with self.assertRaises(ValueError):

            # System dimensions.
            D = 3

            # Drift parameter vector.
            theta = 8.0

            # Diffusion noise parameters vector.
            sigma = 40.0 * np.ones(D)

            # This initialization should fail here.
            _ = Lorenz96(sigma=sigma, theta=theta,
                         dim_D=D, r_seed=911)
        # _end_with_

    # _end_def_

    def test_inverse_sigma(self) -> None:
        """
        Make sure the inverse sigma method
        returns the correct value.

        :return: None
        """

        # Expected inverse sigma.
        expected_vector = 1.0 / self.sigma

        # Make sure they are equal.
        self.assertTrue(np.array_equal(expected_vector,
                                       self.test_obj.inverse_sigma),
                        msg="Inverse Sigma method failed to 7 decimals.")
    # _end_def_

    def test_load_functions(self) -> None:
        """
        Upon initialization the constructor must have loaded
        three files:
            1) one for the Esde,
            2) one for the dEsde_dm,
            3) one for the dEsde_ds.

        NOTE: This tests only if the number of loaded functions
        is the expected one. It does not check the validity of
        the functions.
        """

        # This should be one.
        self.assertTrue(len(self.test_obj.Esde) == 1,
                        msg="The number of loaded energy functions (Esde) is wrong.")

        # This should be one.
        self.assertTrue(len(self.test_obj.dEsde_dm) == 1,
                        msg="The number of loaded gradient functions (dEsde_dm) is wrong.")

        # This should be one.
        self.assertTrue(len(self.test_obj.dEsde_ds) == 1,
                        msg="The number of loaded gradient functions (dEsde_ds) is wrong.")
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
