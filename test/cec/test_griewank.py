import unittest
import numpy as np
from ceco.cec.griewank import Griewank


class Test_griewank(unittest.TestCase):
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
        griewank = Griewank(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = griewank.evaluate(input_vector)
        expected_result = (2**2/4000)+(3**2/4000) - \
            (np.cos(2/np.sqrt(1))*np.cos(3/np.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        griewank = Griewank(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = griewank.evaluate(input_vector)
        expected_result = ((-3)**2/4000)+(2**2/4000) - \
            (np.cos((-3)/np.sqrt(1))*np.cos(2/np.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        griewank = Griewank(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = griewank.evaluate(input_vector)
        expected_result = ((1)**2/4000)+(2**2/4000) - \
            (np.cos((1)/np.sqrt(1))*np.cos(2/np.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        griewank = Griewank(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = griewank.evaluate(input_vector)
        expected_result = (2**2/4000)+((-1)**2/4000) - \
            (np.cos(2/np.sqrt(1))*np.cos((-1)/np.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        griewank = Griewank(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = griewank.evaluate(input_vector)
        expected_result = (0**2/4000)+(0**2/4000) - \
            (np.cos(0/np.sqrt(1))*np.cos(0/np.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        griewank = Griewank(
            self.dimension, self.rotation_identity, self.no_shift, self.f_bias)
        input_vector = np.zeros(2)
        result = griewank.evaluate(input_vector)
        expected_result = (0**2/4000)+(0**2/4000) - \
            (np.cos(0/np.sqrt(1))*np.cos(0/np.sqrt(2)))+1 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
