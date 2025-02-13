import unittest
from ceco.cec.shubert import Shubert
import math


class Test_shubert(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.f_bias = 1.0

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        shubert = Shubert(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = shubert.evaluate(input_vector)
        product = 1.0
        for i in range(2):
            sum_term = 0.0
            for j in range(1, 6):
                sum_term += j * math.cos((j+1)*input_vector[i]+j)
            product *= sum_term
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        shubert = Shubert(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        shifted_vector = [1.0, 2.0]
        result = shubert.evaluate(input_vector)
        product = 1.0
        for i in range(2):
            sum_term = 0.0
            for j in range(1, 6):
                sum_term += j * math.cos((j+1)*shifted_vector[i]+j)
            product *= sum_term
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        shubert = Shubert(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        rotated_vector = [3.0, -2.0]
        result = shubert.evaluate(input_vector)
        product = 1.0
        for i in range(2):
            sum_term = 0.0
            for j in range(1, 6):
                sum_term += j * math.cos((j+1)*rotated_vector[i]+j)
            product *= sum_term
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        shubert = Shubert(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        shifted_rotated_vector = [2.0, -1.0]
        result = shubert.evaluate(input_vector)
        product = 1.0
        for i in range(2):
            sum_term = 0.0
            for j in range(1, 6):
                sum_term += j * math.cos((j+1)*shifted_rotated_vector[i]+j)
            product *= sum_term
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        shubert = Shubert(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]  # Known input
        result = shubert.evaluate(input_vector)
        product = 1.0
        for i in range(2):
            sum_term = 0.0
            for j in range(1, 6):
                sum_term += j * math.cos((j+1)*input_vector[i]+j)
            product *= sum_term
        expected_result = -product
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        shubert = Shubert(self.rotation_identity, self.no_shift,
                          self.f_bias)
        input_vector = [0.0, 0.0]  # Known input
        result = shubert.evaluate(input_vector)
        product = 1.0
        for i in range(2):
            sum_term = 0.0
            for j in range(1, 6):
                sum_term += j * math.cos((j+1)*input_vector[i]+j)
            product *= sum_term
        expected_result = -product + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Shubert(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Shubert(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Shubert(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Shubert(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        shubert = Shubert(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            shubert.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
