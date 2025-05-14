import numpy as np
import unittest
from ceco.cec.different_power import Different_power


class Test_different_power(unittest.TestCase):
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
        different_power = Different_power(self.dimension,
                                          self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = different_power.evaluate(input_vector)
        expected_result = np.abs(2)**(2+4*((1-1)/(2-1))) + \
            np.abs(3)**(2+4*((2-1)/(2-1)))
        expected_result = np.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        different_power = Different_power(
            self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = different_power.evaluate(input_vector)
        expected_result = np.abs(1)**(2+4*((1-1)/(2-1))) + \
            np.abs(2)**(2+4*((2-1)/(2-1)))
        expected_result = np.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        different_power = Different_power(self.dimension,
                                          self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = different_power.evaluate(input_vector)
        expected_result = np.abs(3)**(2+4*((1-1)/(2-1))) + \
            np.abs((-2))**(2+4*((2-1)/(2-1)))
        expected_result = np.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        different_power = Different_power(self.dimension,
                                          self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = different_power.evaluate(input_vector)
        expected_result = np.abs(2)**(2+4*((1-1)/(2-1))) + \
            np.abs((-1))**(2+4*((2-1)/(2-1)))
        expected_result = np.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        different_power = Different_power(self.dimension,
                                          self.rotation_identity, self.no_shift)
        input_vector = np.array([0.0, 0.0])  # Known input
        result = different_power.evaluate(input_vector)
        expected_result = np.abs(0)**(2+4*((1-1)/(2-1))) + \
            np.abs(0)**(2+4*((2-1)/(2-1)))
        expected_result = np.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        different_power = Different_power(self.dimension, self.rotation_identity, self.no_shift,
                                          self.f_bias)
        input_vector = np.array([0.0, 0.0])  # Known input
        result = different_power.evaluate(input_vector)
        expected_result = np.abs(0)**(2+4*((1-1)/(2-1))) + \
            np.abs(0)**(2+4*((2-1)/(2-1)))
        expected_result = np.sqrt(expected_result) + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
