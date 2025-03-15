import unittest
from ceco.cec.shubert import Shubert
import numpy as np


class Test_shubert(unittest.TestCase):
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
        shubert = Shubert(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([3.0, 2.0])
        result = shubert.evaluate(input_vector)
        j_values = np.arange(1, 6)
        cos_terms = np.cos(
            (j_values[:, np.newaxis] + 1) * input_vector + j_values[:, np.newaxis])
        sum_terms = np.sum(j_values[:, np.newaxis] * cos_terms, axis=0)
        product = np.prod(sum_terms)
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        shubert = Shubert(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([3.0, 2.0])
        shifted_vector = np.array([2.0, 1.0])
        result = shubert.evaluate(input_vector)
        j_values = np.arange(1, 6)
        cos_terms = np.cos(
            (j_values[:, np.newaxis] + 1) * shifted_vector + j_values[:, np.newaxis])
        sum_terms = np.sum(j_values[:, np.newaxis] * cos_terms, axis=0)
        product = np.prod(sum_terms)
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        shubert = Shubert(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([3.0, 2.0])
        rotated_vector = np.array([-2.0, 3.0])
        result = shubert.evaluate(input_vector)
        j_values = np.arange(1, 6)
        cos_terms = np.cos(
            (j_values[:, np.newaxis] + 1) * rotated_vector + j_values[:, np.newaxis])
        sum_terms = np.sum(j_values[:, np.newaxis] * cos_terms, axis=0)
        product = np.prod(sum_terms)
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        shubert = Shubert(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([3.0, 2.0])
        shifted_rotated_vector = np.array([-1.0, 2.0])
        result = shubert.evaluate(input_vector)
        j_values = np.arange(1, 6)
        cos_terms = np.cos(
            (j_values[:, np.newaxis] + 1) * shifted_rotated_vector + j_values[:, np.newaxis])
        sum_terms = np.sum(j_values[:, np.newaxis] * cos_terms, axis=0)
        product = np.prod(sum_terms)
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        shubert = Shubert(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = shubert.evaluate(input_vector)
        j_values = np.arange(1, 6)
        cos_terms = np.cos(
            (j_values[:, np.newaxis] + 1) * input_vector + j_values[:, np.newaxis])
        sum_terms = np.sum(j_values[:, np.newaxis] * cos_terms, axis=0)
        product = np.prod(sum_terms)
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        shubert = Shubert(self.dimension, self.rotation_identity,
                          self.no_shift, self.f_bias)
        input_vector = np.zeros(2)
        result = shubert.evaluate(input_vector)
        j_values = np.arange(1, 6)
        cos_terms = np.cos(
            (j_values[:, np.newaxis] + 1) * input_vector + j_values[:, np.newaxis])
        sum_terms = np.sum(j_values[:, np.newaxis] * cos_terms, axis=0)
        product = np.prod(sum_terms)
        expected_result = -product + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
