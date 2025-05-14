import unittest
import numpy as np
from ceco.cec.ackley import Ackley


class Test_ackley(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = np.array([[1, 0], [0, 1]])
        self.rotation_non_identity = np.array(
            [[0, -1], [1, 0]])  # 90-degree rotation
        self.shift = np.array([1, 1])  # Shift vector
        self.no_shift = np.array([0, 0])  # No shift vector
        self.f_bias = 1.0
        self.dimension = 2

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        ackley = Ackley(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = ackley.evaluate(input_vector)
        sum_term1 = np.sum(input_vector**2)
        sum_term2 = np.sum(np.cos(2 * np.pi * input_vector))
        term1 = np.exp(-0.2 * np.sqrt(sum_term1 / 2))
        term2 = -np.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + np.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        ackley = Ackley(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = ackley.evaluate(input_vector)
        shifted_vector = np.array([1.0, 2.0])
        sum_term1 = np.sum(shifted_vector**2)
        sum_term2 = np.sum(np.cos(2 * np.pi * shifted_vector))
        term1 = np.exp(-0.2 * np.sqrt(sum_term1 / 2))
        term2 = -np.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + np.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        ackley = Ackley(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = ackley.evaluate(input_vector)
        rotated_vector = np.array([-3.0, 2.0])
        sum_term1 = np.sum(rotated_vector**2)
        sum_term2 = np.sum(np.cos(2 * np.pi * rotated_vector))
        term1 = np.exp(-0.2 * np.sqrt(sum_term1 / 2))
        term2 = -np.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + np.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        ackley = Ackley(self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = ackley.evaluate(input_vector)
        rotated_vector = np.array([2.0, -1.0])
        sum_term1 = np.sum(rotated_vector**2)
        sum_term2 = np.sum(np.cos(2 * np.pi * rotated_vector))
        term1 = np.exp(-0.2 * np.sqrt(sum_term1 / 2))
        term2 = -np.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + np.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        ackley = Ackley(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([0.0, 0.0])
        result = ackley.evaluate(input_vector)
        sum_term1 = np.sum(input_vector**2)
        sum_term2 = np.sum(np.cos(2 * np.pi * input_vector))
        term1 = np.exp(-0.2 * np.sqrt(sum_term1 / self.dimension))
        term2 = -np.exp(sum_term2 / self.dimension)
        expected_result = -20 * term1 + term2 + 20 + np.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        ackley = Ackley(self.dimension, self.rotation_identity,
                        self.no_shift, self.f_bias)
        input_vector = np.array([0.0, 0.0])
        result = ackley.evaluate(input_vector)
        sum_term1 = np.sum(input_vector**2)
        sum_term2 = np.sum(np.cos(2 * np.pi * input_vector))
        term1 = np.exp(-0.2 * np.sqrt(sum_term1 / self.dimension))
        term2 = -np.exp(sum_term2 / self.dimension)
        expected_result = -20 * term1 + term2 + 20 + np.e + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
