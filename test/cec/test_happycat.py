import unittest
from ceco.cec.happycat import Happycat


class Test_happycat(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.f_bias = 1.0

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        happycat = Happycat(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = happycat.evaluate(input_vector)
        sum_x_squared = sum(z_i ** 2 for z_i in input_vector)
        sum_x = sum(input_vector)

        term1 = abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        happycat = Happycat(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        shifted_vector = [1.0, 2.0]
        result = happycat.evaluate(input_vector)
        sum_x_squared = sum(z_i ** 2 for z_i in shifted_vector)
        sum_x = sum(shifted_vector)

        term1 = abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        happycat = Happycat(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        rotated_vector = [3.0, -2.0]
        result = happycat.evaluate(input_vector)
        sum_x_squared = sum(z_i ** 2 for z_i in rotated_vector)
        sum_x = sum(rotated_vector)

        term1 = abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        happycat = Happycat(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        shifted_rotated_vector = [2.0, -1.0]
        result = happycat.evaluate(input_vector)
        sum_x_squared = sum(z_i ** 2 for z_i in shifted_rotated_vector)
        sum_x = sum(shifted_rotated_vector)

        term1 = abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        happycat = Happycat(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]  # Known input
        result = happycat.evaluate(input_vector)
        sum_x_squared = sum(z_i ** 2 for z_i in input_vector)
        sum_x = sum(input_vector)

        term1 = abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        happycat = Happycat(self.rotation_identity, self.no_shift,
                            self.f_bias)
        input_vector = [0.0, 0.0]  # Known input
        result = happycat.evaluate(input_vector)
        sum_x_squared = sum(z_i ** 2 for z_i in input_vector)
        sum_x = sum(input_vector)

        term1 = abs(sum_x_squared - 2) ** (1/4)
        term2 = (0.5 * sum_x_squared + sum_x) / 2

        expected_result = term1 + term2 + 0.5 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Happycat(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Happycat(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Happycat(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Happycat(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        happycat = Happycat(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            happycat.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
