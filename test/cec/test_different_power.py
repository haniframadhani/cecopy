import math
import unittest
from ceco.cec.different_power import Different_power


class Test_different_power(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.f_bias = 1.0

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        different_power = Different_power(
            self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = different_power.evaluate(input_vector)
        expected_result = abs(2)**(2+4*((1-1)/(2-1))) + \
            abs(3)**(2+4*((2-1)/(2-1)))
        expected_result = math.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        different_power = Different_power(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        result = different_power.evaluate(input_vector)
        expected_result = abs(1)**(2+4*((1-1)/(2-1))) + \
            abs(2)**(2+4*((2-1)/(2-1)))
        expected_result = math.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        different_power = Different_power(
            self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = different_power.evaluate(input_vector)
        expected_result = abs(3)**(2+4*((1-1)/(2-1))) + \
            abs((-2))**(2+4*((2-1)/(2-1)))
        expected_result = math.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        different_power = Different_power(
            self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        result = different_power.evaluate(input_vector)
        expected_result = abs(2)**(2+4*((1-1)/(2-1))) + \
            abs((-1))**(2+4*((2-1)/(2-1)))
        expected_result = math.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        different_power = Different_power(
            self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]  # Known input
        result = different_power.evaluate(input_vector)
        expected_result = abs(0)**(2+4*((1-1)/(2-1))) + \
            abs(0)**(2+4*((2-1)/(2-1)))
        expected_result = math.sqrt(expected_result)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        different_power = Different_power(self.rotation_identity, self.no_shift,
                                          self.f_bias)
        input_vector = [0.0, 0.0]  # Known input
        result = different_power.evaluate(input_vector)
        expected_result = abs(0)**(2+4*((1-1)/(2-1))) + \
            abs(0)**(2+4*((2-1)/(2-1)))
        expected_result = math.sqrt(expected_result) + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Different_power(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Different_power(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Different_power(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Different_power(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        different_power = Different_power(
            rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            different_power.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
