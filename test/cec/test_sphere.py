import unittest
from ceco.cec.sphere import Sphere
import numpy as np


class Test_sphere(unittest.TestCase):
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
        sphere = Sphere(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([3.0, 2.0])
        result = sphere.evaluate(input_vector)
        expected_result = 13.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        sphere = Sphere(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([3.0, 2.0])
        result = sphere.evaluate(input_vector)
        expected_result = 5.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        sphere = Sphere(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([3.0, 2.0])
        result = sphere.evaluate(input_vector)
        expected_result = 13.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        sphere = Sphere(self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([3.0, 2.0])
        result = sphere.evaluate(input_vector)
        expected_result = 5.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        sphere = Sphere(self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = sphere.evaluate(input_vector)
        expected_result = 0.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        sphere = Sphere(self.dimension, self.rotation_identity,
                        self.no_shift, self.f_bias)
        input_vector = np.zeros(2)
        result = sphere.evaluate(input_vector)
        expected_result = 0.0 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
