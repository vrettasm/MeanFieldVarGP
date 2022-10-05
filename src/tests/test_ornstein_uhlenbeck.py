import unittest

# Import custom code
from dynamical_systems.ornstein_uhlenbeck import OrnsteinUhlenbeck


class TestOrnsteinUhlenbeck(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> Testing OrnsteinUhlenbeck - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> Testing OrnsteinUhlenbeck - STOP -", end="\n\n")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with fixed values.

        :return: None.
        """

        # Create an object with fixed input values.
        self.test_obj = OrnsteinUhlenbeck(sigma=1.0, theta=[2.0, 0.0], r_seed=911)
    # _end_def_

    def test_inverse_sigma(self) -> None:
        """
        Ensure the inverse sigma method returns the correct value.

        :return: None
        """

        # Make sure they are equal (to seven decimals).
        self.assertAlmostEqual(float(1.0/1.0),
                               self.test_obj.inverse_sigma,
                               msg="Inverse Sigma method failed to 7 decimals.")
    # _end_def_

    def test_load_functions(self) -> None:
        """
        Upon initialization the constructor must have loaded 3 files:

            1) one for the Esde,
            2) one for the dEsde_dm,
            3) one for the dEsde_ds.

        NOTE: This tests only if the number of loaded functions is the
        expected one. It does not check the validity of the functions.
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
