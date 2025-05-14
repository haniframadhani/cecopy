import unittest
import numpy as np
from ceco.cec.rastrigin import Rastrigin


class Test_ackley(unittest.TestCase):
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
        rastrigin = Rastrigin(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = rastrigin.evaluate(input_vector)
        expected_result = (2**2-10*np.cos(2*np.pi*2)+10) + \
            (3**2-10*np.cos(2*np.pi*3)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        rastrigin = Rastrigin(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = rastrigin.evaluate(input_vector)
        expected_result = ((-3)**2-10*np.cos(2*np.pi*(-3))+10) + \
            (2**2-10*np.cos(2*np.pi*2)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        rastrigin = Rastrigin(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = rastrigin.evaluate(input_vector)
        expected_result = (2**2-10*np.cos(2*np.pi*2)+10) + \
            ((-1)**2-10*np.cos(2*np.pi*(-1))+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        rastrigin = Rastrigin(
            self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = rastrigin.evaluate(input_vector)
        expected_result = (1**2-10*np.cos(2*np.pi*1)+10) + \
            (2**2-10*np.cos(2*np.pi*2)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        rastrigin = Rastrigin(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = rastrigin.evaluate(input_vector)
        expected_result = (0**2-10*np.cos(2*np.pi*0)+10) + \
            (0**2-10*np.cos(2*np.pi*0)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        rastrigin = Rastrigin(
            self.dimension, self.rotation_identity, self.no_shift, self.f_bias)
        input_vector = np.zeros(2)
        result = rastrigin.evaluate(input_vector)
        expected_result = (0**2-10*np.cos(2*np.pi*0)+10) + \
            (0**2-10*np.cos(2*np.pi*0)+10)+self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
