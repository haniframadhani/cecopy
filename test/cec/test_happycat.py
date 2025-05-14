import unittest
from ceco.cec.happycat import Happycat
import numpy as np


class Test_happycat(unittest.TestCase):
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
        happycat = Happycat(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = happycat.evaluate(input_vector)
        sum_x_squared = np.sum(input_vector ** 2)
        sum_x = np.sum(input_vector)

        term1 = np.abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        happycat = Happycat(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        shifted_vector = np.array([1.0, 2.0])
        result = happycat.evaluate(input_vector)
        sum_x_squared = np.sum(shifted_vector ** 2)
        sum_x = np.sum(shifted_vector)

        term1 = np.abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        happycat = Happycat(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        rotated_vector = np.array([-3.0, 2.0])
        result = happycat.evaluate(input_vector)
        sum_x_squared = np.sum(rotated_vector ** 2)
        sum_x = np.sum(rotated_vector)

        term1 = np.abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        happycat = Happycat(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        shifted_rotated_vector = np.array([-2, 1])
        result = happycat.evaluate(input_vector)
        sum_x_squared = np.sum(shifted_rotated_vector ** 2)
        sum_x = np.sum(shifted_rotated_vector)

        term1 = np.abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        happycat = Happycat(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = happycat.evaluate(input_vector)
        sum_x_squared = np.sum(input_vector ** 2)
        sum_x = np.sum(input_vector)

        term1 = np.abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        happycat = Happycat(self.dimension, self.rotation_identity, self.no_shift,
                            self.f_bias)
        input_vector = np.zeros(2)
        result = happycat.evaluate(input_vector)
        sum_x_squared = np.sum(input_vector ** 2)
        sum_x = np.sum(input_vector)

        term1 = np.abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
