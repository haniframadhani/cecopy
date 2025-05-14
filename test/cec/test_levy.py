import unittest
from ceco.cec.levy import Levy
import numpy as np


class Test_levy(unittest.TestCase):
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
        levy = Levy(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = levy.evaluate(input_vector)
        w = np.array([1.25, 1.5])
        term1 = np.sin(np.pi * w[0]) ** 2

        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 *
                          np.sin(np.pi * w[:-1] + 1) ** 2))
        term2 = (w[-1] - 1) ** 2 * \
            (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        levy = Levy(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = levy.evaluate(input_vector)
        w = np.array([1.0, 1.25])
        term1 = np.sin(np.pi * w[0]) ** 2

        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 *
                          np.sin(np.pi * w[:-1] + 1) ** 2))
        term2 = (w[-1] - 1) ** 2 * \
            (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        levy = Levy(self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        w = np.array([0, 1.25])
        result = levy.evaluate(input_vector)
        term1 = np.sin(np.pi * w[0]) ** 2

        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 *
                          np.sin(np.pi * w[:-1] + 1) ** 2))
        term2 = (w[-1] - 1) ** 2 * \
            (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        levy = Levy(self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = levy.evaluate(input_vector)
        w = np.array([0.25, 1])
        term1 = np.sin(np.pi * w[0]) ** 2

        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 *
                          np.sin(np.pi * w[:-1] + 1) ** 2))
        term2 = (w[-1] - 1) ** 2 * \
            (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        levy = Levy(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = levy.evaluate(input_vector)
        w = np.array([0.75, 0.75])
        term1 = np.sin(np.pi * w[0]) ** 2

        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 *
                          np.sin(np.pi * w[:-1] + 1) ** 2))
        term2 = (w[-1] - 1) ** 2 * \
            (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        levy = Levy(self.dimension, self.rotation_identity, self.no_shift,
                    self.f_bias)
        input_vector = np.zeros(2)
        result = levy.evaluate(input_vector)
        w = np.array([0.75, 0.75])
        term1 = np.sin(np.pi * w[0]) ** 2

        sum_term = np.sum((w[:-1] - 1) ** 2 * (1 + 10 *
                          np.sin(np.pi * w[:-1] + 1) ** 2))
        term2 = (w[-1] - 1) ** 2 * \
            (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
