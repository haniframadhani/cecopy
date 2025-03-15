import unittest
from ceco.cec.vincent import Vincent
import numpy as np


class Test_vincent(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = np.eye(2)
        self.rotation_non_identity = np.array([[0, -1], [1, 0]])
        self.shift = np.array([1, 1])
        self.no_shift = np.zeros(2)
        self.f_bias = 1.0
        self.dimension = 2

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        vincent = Vincent(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([3.0, 2.0])
        result = vincent.evaluate(input_vector)
        expected_result = -0.1980669543
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        vincent = Vincent(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([3.0, 2.0])
        result = vincent.evaluate(input_vector)
        expected_result = 0.3019107135584
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        vincent = Vincent(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, -3.0])
        result = vincent.evaluate(input_vector)
        expected_result = -0.19806695436314
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        vincent = Vincent(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([3.0, -2.0])
        result = vincent.evaluate(input_vector)
        expected_result = -0.1980669543
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        vincent = Vincent(self.dimension, self.rotation_identity,
                          self.no_shift, self.f_bias)
        input_vector = np.array([3.0, 2.0])
        result = vincent.evaluate(input_vector)
        expected_result = (-0.1980669543) + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        with self.assertRaises(ValueError) as context:
            vincent = Vincent(self.dimension, self.rotation_identity,
                              self.no_shift, self.f_bias)
            input_vector = np.zeros(2)
            result = vincent.evaluate(input_vector)
        self.assertEqual(str(context.exception),
                         "All elements in input_vector must be greater than 0 after shift and rotation")


if __name__ == '__main__':
    unittest.main()
