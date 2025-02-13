import unittest
from ceco.cec.levy import Levy
import math


class Test_levy(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.f_bias = 1.0

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        levy = Levy(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        w = [1.25, 1.5]
        result = levy.evaluate(input_vector)
        term1 = math.sin(math.pi * w[0]) ** 2

        sum_term = 0.0
        for i in range(1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 *
                                           math.sin(math.pi * w[i] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * \
            (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        levy = Levy(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        # shifted_vector = [1.0, 2.0]
        w = [1.0, 1.25]
        result = levy.evaluate(input_vector)
        term1 = math.sin(math.pi * w[0]) ** 2

        sum_term = 0.0
        for i in range(1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 *
                                           math.sin(math.pi * w[i] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * \
            (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        levy = Levy(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        # rotated_vector = [3.0, -2.0]
        w = [1.5, 0.25]
        result = levy.evaluate(input_vector)
        term1 = math.sin(math.pi * w[0]) ** 2

        sum_term = 0.0
        for i in range(1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 *
                                           math.sin(math.pi * w[i] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * \
            (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        levy = Levy(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        # shifted_rotated_vector = [2.0, -1.0]
        w = [1.25, 0.5]
        result = levy.evaluate(input_vector)
        term1 = math.sin(math.pi * w[0]) ** 2

        sum_term = 0.0
        for i in range(1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 *
                                           math.sin(math.pi * w[i] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * \
            (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        levy = Levy(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]  # Known input
        w = [0.75, 0.75]
        result = levy.evaluate(input_vector)
        term1 = math.sin(math.pi * w[0]) ** 2

        sum_term = 0.0
        for i in range(1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 *
                                           math.sin(math.pi * w[i] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * \
            (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        levy = Levy(self.rotation_identity, self.no_shift,
                    self.f_bias)
        input_vector = [0.0, 0.0]  # Known input
        w = [0.75, 0.75]
        result = levy.evaluate(input_vector)
        term1 = math.sin(math.pi * w[0]) ** 2

        sum_term = 0.0
        for i in range(1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 *
                                           math.sin(math.pi * w[i] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * \
            (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        expected_result = term1+sum_term+term2 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Levy(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Levy(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Levy(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Levy(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        levy = Levy(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            levy.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
