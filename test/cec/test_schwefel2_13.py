import unittest
from ceco.cec.schwefel2_13 import Schwefel2_13
import math


class Test_schwefel2_13(unittest.TestCase):
    def setUp(self):
        # Define a simple rotation matrix and shift vector for testing
        self.rotation = [[1, 0], [0, 1]]  # Identity matrix
        self.shift = [0, 0]  # No shift
        self.schwefel = Schwefel2_13(self.rotation, self.shift)

    def test_generate_random_matrix(self):
        D = 2
        min_val = -100
        max_val = 100
        matrix = self.schwefel.generate_random_matrix(D, min_val, max_val)

        # Check the dimensions of the matrix
        self.assertEqual(len(matrix), D)
        self.assertEqual(len(matrix[0]), D)

        # Check that all values are within the specified range
        for row in matrix:
            for val in row:
                self.assertTrue(min_val <= val <= max_val)

    def test_generate_random_alpha(self):
        D = 2
        alpha = self.schwefel.generate_random_alpha(D)

        # Check the length of the alpha vector
        self.assertEqual(len(alpha), D)

        # Check that all values are within the range [-π, π]
        for val in alpha:
            self.assertTrue(-math.pi <= val <= math.pi)

    def test_compute_A(self):
        D = 2
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        alpha = [math.pi / 2, math.pi / 4]

        A = self.schwefel.compute_A(D, a, b, alpha)

        # Expected A values calculated manually
        expected_A = [
            a[0][0] * math.sin(alpha[0]) + a[0][1] * math.sin(alpha[1]) +
            b[0][0] * math.cos(alpha[0]) + b[0][1] * math.cos(alpha[1]),
            a[1][0] * math.sin(alpha[0]) + a[1][1] * math.sin(alpha[1]) +
            b[1][0] * math.cos(alpha[0]) + b[1][1] * math.cos(alpha[1])
        ]

        self.assertEqual(len(A), D)
        for i in range(D):
            self.assertAlmostEqual(A[i], expected_A[i])

    def test_compute_B(self):
        D = 2
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        X = [math.pi / 2, math.pi / 4]

        B = self.schwefel.compute_B(D, a, b, X)

        # Expected B values calculated manually
        expected_B = [
            a[0][0] * math.sin(X[0]) + a[0][1] * math.sin(X[1]) +
            b[0][0] * math.cos(X[0]) + b[0][1] * math.cos(X[1]),
            a[1][0] * math.sin(X[0]) + a[1][1] * math.sin(X[1]) +
            b[1][0] * math.cos(X[0]) + b[1][1] * math.cos(X[1])
        ]

        self.assertEqual(len(B), D)
        for i in range(D):
            self.assertAlmostEqual(B[i], expected_B[i])

    def test_evaluate(self):
        input_vector = [math.pi / 2, math.pi / 4]

        # Since the matrices a and b are generated randomly, we can't predict the exact result.
        # Instead, we can check that the result is a non-negative float.
        result = self.schwefel.evaluate(input_vector)

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_evaluate_with_rotation_and_shift(self):
        # Define a non-identity rotation matrix and a non-zero shift vector
        rotation = [[0, 1], [1, 0]]  # Swap x and y
        shift = [1, 2]
        schwefel = Schwefel2_13(rotation, shift)

        input_vector = [math.pi / 2, math.pi / 4]

        # Rotated and shifted vector should be [input_vector[1] - shift[0], input_vector[0] - shift[1]]
        rotated_shifted_vector = [input_vector[1] -
                                  shift[0], input_vector[0] - shift[1]]

        # Since the matrices a and b are generated randomly, we can't predict the exact result.
        # Instead, we can check that the result is a non-negative float.
        result = schwefel.evaluate(input_vector)

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)


if __name__ == '__main__':
    unittest.main()
