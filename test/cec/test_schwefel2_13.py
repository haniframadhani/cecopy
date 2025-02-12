import unittest
from ceco.cec.schwefel2_13 import Schwefel2_13
import math


class Test_schwefel2_13(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.static_a = [[1, 2], [3, 4]]
        self.static_b = [[5, 6], [7, 8]]
        self.static_alpha = [0, 0]
        self.f_bias = 1.0

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        schwefel2_13 = Schwefel2_13(self.rotation_identity, self.no_shift, 2)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = [3.0, 2.0]
        result = schwefel2_13.evaluate(input_vector)
        A = [0.0] * 2
        for i in range(2):
            for j in range(2):
                A[i] += schwefel2_13.a[i][j] * math.sin(schwefel2_13.alpha[j]) + \
                    schwefel2_13.b[i][j] * math.cos(schwefel2_13.alpha[j])
        B = [0.0] * 2
        for i in range(2):
            for j in range(2):
                B[i] += schwefel2_13.a[i][j] * \
                    math.sin(
                        input_vector[j]) + schwefel2_13.b[i][j] * math.cos(input_vector[j])
        expected_result = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        schwefel2_13 = Schwefel2_13(self.rotation_identity, self.shift, 2)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = [3.0, 2.0]
        result = schwefel2_13.evaluate(input_vector)
        shift_vector = [2.0, 1.0]
        A = [0.0] * 2
        for i in range(2):
            for j in range(2):
                A[i] += schwefel2_13.a[i][j] * math.sin(schwefel2_13.alpha[j]) + \
                    schwefel2_13.b[i][j] * math.cos(schwefel2_13.alpha[j])
        B = [0.0] * 2
        for i in range(2):
            for j in range(2):
                B[i] += schwefel2_13.a[i][j] * \
                    math.sin(
                        shift_vector[j]) + schwefel2_13.b[i][j] * math.cos(shift_vector[j])
        expected_result = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        schwefel2_13 = Schwefel2_13(
            self.rotation_non_identity, self.no_shift, 2)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = [3.0, 2.0]
        result = schwefel2_13.evaluate(input_vector)
        rotated_vector = [2.0, -3.0]
        A = [0.0] * 2
        for i in range(2):
            for j in range(2):
                A[i] += schwefel2_13.a[i][j] * math.sin(schwefel2_13.alpha[j]) + \
                    schwefel2_13.b[i][j] * math.cos(schwefel2_13.alpha[j])
        B = [0.0] * 2
        for i in range(2):
            for j in range(2):
                B[i] += schwefel2_13.a[i][j] * \
                    math.sin(
                        rotated_vector[j]) + schwefel2_13.b[i][j] * math.cos(rotated_vector[j])
        expected_result = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        schwefel2_13 = Schwefel2_13(self.rotation_non_identity, self.shift, 2)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = [3.0, 2.0]
        result = schwefel2_13.evaluate(input_vector)
        Z = [1.0, -2.0]
        A = [0.0] * 2
        for i in range(2):
            for j in range(2):
                A[i] += schwefel2_13.a[i][j] * math.sin(schwefel2_13.alpha[j]) + \
                    schwefel2_13.b[i][j] * math.cos(schwefel2_13.alpha[j])
        B = [0.0] * 2
        for i in range(2):
            for j in range(2):
                B[i] += schwefel2_13.a[i][j] * \
                    math.sin(
                        Z[j]) + schwefel2_13.b[i][j] * math.cos(Z[j])
        expected_result = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        schwefel2_13 = Schwefel2_13(self.rotation_identity, self.no_shift, 2)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = [0.0, 0.0]  # Known input
        result = schwefel2_13.evaluate(input_vector)
        A = [0.0] * 2
        for i in range(2):
            for j in range(2):
                A[i] += schwefel2_13.a[i][j] * math.sin(schwefel2_13.alpha[j]) + \
                    schwefel2_13.b[i][j] * math.cos(schwefel2_13.alpha[j])
        B = [0.0] * 2
        for i in range(2):
            for j in range(2):
                B[i] += schwefel2_13.a[i][j] * \
                    math.sin(
                        input_vector[j]) + schwefel2_13.b[i][j] * math.cos(input_vector[j])
        expected_result = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        schwefel2_13 = Schwefel2_13(
            self.rotation_identity, self.no_shift, 2, self.f_bias)
        schwefel2_13.a = self.static_a
        schwefel2_13.b = self.static_b
        schwefel2_13.alpha = self.static_alpha
        input_vector = [0.0, 0.0]  # Known input
        result = schwefel2_13.evaluate(input_vector)
        A = [0.0] * 2
        for i in range(2):
            for j in range(2):
                A[i] += schwefel2_13.a[i][j] * math.sin(schwefel2_13.alpha[j]) + \
                    schwefel2_13.b[i][j] * math.cos(schwefel2_13.alpha[j])
        B = [0.0] * 2
        for i in range(2):
            for j in range(2):
                B[i] += schwefel2_13.a[i][j] * \
                    math.sin(
                        input_vector[j]) + schwefel2_13.b[i][j] * math.cos(input_vector[j])
        expected_result = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Schwefel2_13(rotation="invalid", shift=[0, 0], dimension=2)
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Schwefel2_13(rotation=[], shift=[0, 0], dimension=2)
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Schwefel2_13(rotation=[[1, 2, 3], [4, 5, 6]],
                         shift=[0, 0, 0], dimension=3)
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Schwefel2_13(rotation=[[1, 0], [0, 1]],
                         shift=[0, 0, 0], dimension=2)
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        schwefel2_13 = Schwefel2_13(
            rotation=[[1, 0], [0, 1]], shift=[0, 0], dimension=2)
        with self.assertRaises(ValueError) as context:
            schwefel2_13.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
