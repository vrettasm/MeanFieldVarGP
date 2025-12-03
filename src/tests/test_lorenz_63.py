import unittest

import numpy as np

# Import custom code
from src.dynamical_systems.lorenz_63 import Lorenz63


class TestLorenz63(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> Testing Lorenz63 - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> Testing Lorenz63 - STOP -", end="\n\n")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with fixed values.

        :return: None.
        """

        # Create an object with fixed input values.
        self.test_obj = Lorenz63(sigma=[10.0, 20.0, 30.0],
                                 theta=[10.0, 28.0, 2.67],
                                 r_seed=911)
    # _end_def_

    def test_inverse_sigma(self) -> None:
        """
        Ensure the inverse sigma method returns the correct value.

        :return: None
        """

        # Make sure they are equal.
        self.assertTrue(np.array_equal(1.0/np.asarray([10.0, 20.0, 30.0]),
                                       self.test_obj.inverse_sigma),
                        msg="Inverse Sigma method failed to 7 decimals.")
    # _end_def_

    def test_load_functions(self) -> None:
        """
        Upon initialization the constructor must have loaded 9 files:

            1) three for the Esde,
            2) three for the dEsde_dm,
            3) three for the dEsde_ds.

        NOTE: This tests only if the number of loaded functions is the
        expected one. It does not check the validity of the functions.
        """

        # This should be one.
        self.assertTrue(len(self.test_obj.Esde) == 3,
                        msg="The number of loaded energy functions (Esde) is wrong.")

        # This should be one.
        self.assertTrue(len(self.test_obj.dEsde_dm) == 3,
                        msg="The number of loaded gradient functions (dEsde_dm) is wrong.")

        # This should be one.
        self.assertTrue(len(self.test_obj.dEsde_ds) == 3,
                        msg="The number of loaded gradient functions (dEsde_ds) is wrong.")
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
