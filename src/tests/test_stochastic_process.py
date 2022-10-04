import unittest

import numpy as np

# Import custom code
from src.dynamical_systems.stochastic_process import StochasticProcess


class TestStochasticProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> Testing StochasticProcess - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> Testing StochasticProcess - STOP -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with fixed rng value.

        :return: None.
        """

        # Create an object with fixed RNG.
        self.test_obj = StochasticProcess(r_seed=911)
    # _end_def_

    def test_initializations(self) -> None:
        """
        Test the object for un-initialized values.

        :return: None
        """

        with self.assertRaises(NotImplementedError):

            # Get the theta vector.
            _ = self.test_obj.theta

        # _end_with_

        with self.assertRaises(NotImplementedError):

            # Get the sigma vector.
            _ = self.test_obj.sigma

        # _end_with_

        with self.assertRaises(NotImplementedError):

            # Get the sample path.
            _ = self.test_obj.sample_path

        # _end_with_

        with self.assertRaises(NotImplementedError):

            # Get the discrete time window.
            _ = self.test_obj.time_window

        # _end_with_

        with self.assertRaises(NotImplementedError):

            # Get the discrete time step.
            _ = self.test_obj.time_step

        # _end_with_

    # _end_def_

    def test_rng_seed(self) -> None:
        """
        Make sure the input to the 'RNG' is set properly.
        The seed value (911) corresponds to the following
        state values:

        'state': 113219229968428971613983651843996252106.

        :return: None
        """

        # Get the object state.
        state = int(self.test_obj.rng.bit_generator.state["state"]["state"])

        # This should be equal.
        self.assertEqual(int('113219229968428971613983651843996252106'),
                         state)
    # _end_def_

    def test_sigma_accessor(self) -> None:
        """
        Make sure the sigma accessor does not accept zeros
        or negative values. It represents variance, hence
        it should only contain positive values.

        :return: None
        """

        # Because the order of the execution should not matter
        # we can't use the self.test_obj here, to avoid failing
        # the other tests.
        local_obj = StochasticProcess()

        # Negative variance should not be allowed.
        with self.assertRaises(ValueError):
            local_obj.sigma = -1.0
        # _end_with_

        # Negative and zero variances should not be allowed.
        with self.assertRaises(ValueError):
            local_obj.sigma = [0.0, -1.0]
        # _end_with_

        # Zero variance should not be allowed.
        with self.assertRaises(ValueError):
            local_obj.sigma = [10.0, 0.0]
        # _end_with_

        # This assignment should be accepted.
        local_obj.sigma = [10.0, 10.0]

        # Make sure they are set properly.
        self.assertTrue(np.array_equal(np.array([10.0, 10.0]),
                                       local_obj.sigma))
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
