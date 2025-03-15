import unittest
import numpy as np
from ceco.cec.weierstrass import Weierstrass


class Test_weierstrass(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = np.eye(2)
        self.rotation_non_identity = np.array([[0, -1], [1, 0]])
        self.shift = np.array([1, 1])
        self.no_shift = np.zeros(2)
        self.f_bias = 1.0
        self.dimension = 2
        self.a = 0.5
        self.b = 3
        self.k_max = 20

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        weierstrass = Weierstrass(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([3.0, 2.0])
        result = weierstrass.evaluate(input_vector)
        k_values = np.arange(self.k_max + 1)
        a_pow_k = self.a ** k_values
        b_pow_k = self.b ** k_values

        inner_sum = np.sum(
            a_pow_k[:, np.newaxis] *
            np.cos(2 * np.pi * b_pow_k[:, np.newaxis]
                   * (input_vector + 0.5)),
            axis=0
        )
        total_sum = np.sum(inner_sum)

        second_sum = np.sum(
            a_pow_k * np.cos(2 * np.pi * b_pow_k * 0.5)
        )

        expected_result = total_sum - self.dimension * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        weierstrass = Weierstrass(
            self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([3.0, 2.0])
        result = weierstrass.evaluate(input_vector)
        shift_vector = np.array([2.0, 1.0])
        k_values = np.arange(self.k_max + 1)
        a_pow_k = self.a ** k_values
        b_pow_k = self.b ** k_values

        inner_sum = np.sum(
            a_pow_k[:, np.newaxis] *
            np.cos(2 * np.pi * b_pow_k[:, np.newaxis]
                   * (shift_vector + 0.5)),
            axis=0
        )
        total_sum = np.sum(inner_sum)

        second_sum = np.sum(
            a_pow_k * np.cos(2 * np.pi * b_pow_k * 0.5)
        )

        expected_result = total_sum - self.dimension * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        weierstrass = Weierstrass(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([3.0, 2.0])
        result = weierstrass.evaluate(input_vector)
        z = np.array([-1.0, 2.0])
        k_values = np.arange(self.k_max + 1)
        a_pow_k = self.a ** k_values
        b_pow_k = self.b ** k_values

        inner_sum = np.sum(
            a_pow_k[:, np.newaxis] *
            np.cos(2 * np.pi * b_pow_k[:, np.newaxis]
                   * (z + 0.5)),
            axis=0
        )
        total_sum = np.sum(inner_sum)

        second_sum = np.sum(
            a_pow_k * np.cos(2 * np.pi * b_pow_k * 0.5)
        )

        expected_result = total_sum - self.dimension * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        weierstrass = Weierstrass(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([3.0, 2.0])
        result = weierstrass.evaluate(input_vector)
        rotated_vector = np.array([-2.0, 3.0])
        k_values = np.arange(self.k_max + 1)
        a_pow_k = self.a ** k_values
        b_pow_k = self.b ** k_values

        inner_sum = np.sum(
            a_pow_k[:, np.newaxis] *
            np.cos(2 * np.pi * b_pow_k[:, np.newaxis]
                   * (rotated_vector + 0.5)),
            axis=0
        )
        total_sum = np.sum(inner_sum)

        second_sum = np.sum(
            a_pow_k * np.cos(2 * np.pi * b_pow_k * 0.5)
        )

        expected_result = total_sum - self.dimension * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        weierstrass = Weierstrass(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = weierstrass.evaluate(input_vector)
        k_values = np.arange(self.k_max + 1)
        a_pow_k = self.a ** k_values
        b_pow_k = self.b ** k_values

        inner_sum = np.sum(
            a_pow_k[:, np.newaxis] *
            np.cos(2 * np.pi * b_pow_k[:, np.newaxis]
                   * (input_vector + 0.5)),
            axis=0
        )
        total_sum = np.sum(inner_sum)

        second_sum = np.sum(
            a_pow_k * np.cos(2 * np.pi * b_pow_k * 0.5)
        )

        expected_result = total_sum - self.dimension * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        weierstrass = Weierstrass(
            self.dimension, self.rotation_identity, self.no_shift, self.f_bias)
        input_vector = np.zeros(2)
        result = weierstrass.evaluate(input_vector)
        k_values = np.arange(self.k_max + 1)
        a_pow_k = self.a ** k_values
        b_pow_k = self.b ** k_values

        inner_sum = np.sum(
            a_pow_k[:, np.newaxis] *
            np.cos(2 * np.pi * b_pow_k[:, np.newaxis]
                   * (input_vector + 0.5)),
            axis=0
        )
        total_sum = np.sum(inner_sum)

        second_sum = np.sum(
            a_pow_k * np.cos(2 * np.pi * b_pow_k * 0.5)
        )

        expected_result = total_sum - self.dimension * second_sum + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
