import unittest
import numpy as np
from ceco.cec.discus import Discus


class Test_discus(unittest.TestCase):
    def setUp(self):
        self.rotation_identity = np.eye(2)
        self.rotation_non_identity = np.array([[0, -1], [1, 0]])
        self.shift = np.array([1, 1])
        self.no_shift = np.zeros(2)
        self.f_bias = 1.0
        self.dimension = 2

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        discus = Discus(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = discus.evaluate(input_vector)
        expected_result = 4_000_009
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        discus = Discus(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = discus.evaluate(input_vector)
        expected_result = 1_000_004
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        discus = Discus(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = discus.evaluate(input_vector)
        expected_result = 9_000_004
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        discus = Discus(self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = discus.evaluate(input_vector)
        expected_result = 4_000_001
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        discus = Discus(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = discus.evaluate(input_vector)
        expected_result = 0.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        discus = Discus(self.dimension, self.rotation_identity,
                        self.no_shift, self.f_bias)
        input_vector = np.zeros(2)
        result = discus.evaluate(input_vector)
        expected_result = 1.0
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
