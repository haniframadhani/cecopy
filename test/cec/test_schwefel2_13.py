import unittest
from ceco.cec.schwefel2_13 import Schwefel2_13
import numpy as np


class Test_schwefel2_13(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = np.eye(2)
        self.rotation_non_identity = np.array([[0, -1], [1, 0]])
        self.shift = np.array([1, 1])
        self.no_shift = np.zeros(2)
        self.f_bias = 1.0
        self.dimension = 2
        self.static_a = np.array([[1, 2], [3, 4]])
        self.static_b = np.array([[5, 6], [7, 8]])
        self.static_alpha = np.zeros(2)

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        schwefel2_13 = Schwefel2_13(
            self.dimension, self.rotation_identity, self.no_shift)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = np.array([3.0, 2.0])
        result = schwefel2_13.evaluate(input_vector)
        A = np.sum(
            schwefel2_13.a * np.sin(schwefel2_13.alpha) +
            schwefel2_13.b * np.cos(schwefel2_13.alpha),
            axis=1
        )
        B = np.sum(
            schwefel2_13.a * np.sin(input_vector) +
            schwefel2_13.b * np.cos(input_vector),
            axis=1
        )
        expected_result = np.sum((A - B) ** 2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        schwefel2_13 = Schwefel2_13(
            self.dimension, self.rotation_identity, self.shift)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = np.array([3.0, 2.0])
        result = schwefel2_13.evaluate(input_vector)
        shift_vector = np.array([2.0, 1.0])
        A = np.sum(
            schwefel2_13.a * np.sin(schwefel2_13.alpha) +
            schwefel2_13.b * np.cos(schwefel2_13.alpha),
            axis=1
        )
        B = np.sum(
            schwefel2_13.a * np.sin(shift_vector) +
            schwefel2_13.b * np.cos(shift_vector),
            axis=1
        )
        expected_result = np.sum((A - B) ** 2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        schwefel2_13 = Schwefel2_13(
            self.dimension, self.rotation_non_identity, self.no_shift)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = np.array([3.0, 2.0])
        result = schwefel2_13.evaluate(input_vector)
        rotated_vector = np.array([-2.0, 3.0])
        A = np.sum(
            schwefel2_13.a * np.sin(schwefel2_13.alpha) +
            schwefel2_13.b * np.cos(schwefel2_13.alpha),
            axis=1
        )
        B = np.sum(
            schwefel2_13.a * np.sin(rotated_vector) +
            schwefel2_13.b * np.cos(rotated_vector),
            axis=1
        )
        expected_result = np.sum((A - B) ** 2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        schwefel2_13 = Schwefel2_13(
            self.dimension, self.rotation_non_identity, self.shift)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = np.array([3.0, 2.0])
        result = schwefel2_13.evaluate(input_vector)
        Z = np.array([-1.0, 2.0])
        A = np.sum(
            schwefel2_13.a * np.sin(schwefel2_13.alpha) +
            schwefel2_13.b * np.cos(schwefel2_13.alpha),
            axis=1
        )
        B = np.sum(
            schwefel2_13.a * np.sin(Z) +
            schwefel2_13.b * np.cos(Z),
            axis=1
        )
        expected_result = np.sum((A - B) ** 2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        schwefel2_13 = Schwefel2_13(
            self.dimension, self.rotation_identity, self.no_shift)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = np.zeros(2)
        result = schwefel2_13.evaluate(input_vector)
        A = np.sum(
            schwefel2_13.a * np.sin(schwefel2_13.alpha) +
            schwefel2_13.b * np.cos(schwefel2_13.alpha),
            axis=1
        )
        B = np.sum(
            schwefel2_13.a * np.sin(input_vector) +
            schwefel2_13.b * np.cos(input_vector),
            axis=1
        )
        expected_result = np.sum((A - B) ** 2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        schwefel2_13 = Schwefel2_13(
            self.dimension, self.rotation_identity, self.no_shift, self.f_bias)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = np.zeros(2)
        result = schwefel2_13.evaluate(input_vector)
        A = np.sum(
            schwefel2_13.a * np.sin(schwefel2_13.alpha) +
            schwefel2_13.b * np.cos(schwefel2_13.alpha),
            axis=1
        )
        B = np.sum(
            schwefel2_13.a * np.sin(input_vector) +
            schwefel2_13.b * np.cos(input_vector),
            axis=1
        )
        expected_result = np.sum((A - B) ** 2) + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
