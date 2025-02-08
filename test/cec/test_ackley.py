import unittest
import math
from ceco.cec.ackley import Ackley


class Test_ackley(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        ackley = Ackley(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]
        result = ackley.evaluate(input_vector)
        sum_term1 = sum(z_i ** 2 for z_i in input_vector)
        sum_term2 = sum(math.cos(2 * math.pi * z_i) for z_i in input_vector)
        term1 = math.exp(-0.2 * math.sqrt(sum_term1 / 2))
        term2 = -math.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + math.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        ackley = Ackley(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = ackley.evaluate(input_vector)
        shifted_vector = [1.0, 2.0]
        sum_term1 = sum(z_i ** 2 for z_i in shifted_vector)
        sum_term2 = sum(math.cos(2 * math.pi * z_i) for z_i in shifted_vector)
        term1 = math.exp(-0.2 * math.sqrt(sum_term1 / 2))
        term2 = -math.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + math.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        ackley = Ackley(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]
        result = ackley.evaluate(input_vector)
        rotated_vector = [-3.0, 2.0]
        sum_term1 = sum(z_i ** 2 for z_i in rotated_vector)
        sum_term2 = sum(math.cos(2 * math.pi * z_i) for z_i in rotated_vector)
        term1 = math.exp(-0.2 * math.sqrt(sum_term1 / 2))
        term2 = -math.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + math.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        ackley = Ackley(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = ackley.evaluate(input_vector)
        rotated_vector = [2.0, -1.0]
        sum_term1 = sum(z_i ** 2 for z_i in rotated_vector)
        sum_term2 = sum(math.cos(2 * math.pi * z_i) for z_i in rotated_vector)
        term1 = math.exp(-0.2 * math.sqrt(sum_term1 / 2))
        term2 = -math.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + math.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        ackley = Ackley(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]
        result = ackley.evaluate(input_vector)
        sum_term1 = sum(z_i ** 2 for z_i in input_vector)
        sum_term2 = sum(math.cos(2 * math.pi * z_i) for z_i in input_vector)
        term1 = math.exp(-0.2 * math.sqrt(sum_term1 / 2))
        term2 = -math.exp(sum_term2 / 2)
        expected_result = -20 * term1 + term2 + 20 + math.e
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Ackley(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Ackley(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Ackley(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Ackley(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        ackley = Ackley(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            ackley.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
