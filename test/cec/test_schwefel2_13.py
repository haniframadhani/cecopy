import unittest
from ceco.cec.schwefel2_13 import Schwefel2_13


class Test_schwefel2_13(unittest.TestCase):
    def setUp(self):
        # Static rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.static_a = [[1, 2], [3, 4]]
        self.static_b = [[5, 6], [7, 8]]
        self.static_alpha = [0, 0]

    def test_evaluate_with_shift_and_rotation(self):
        schwefel = Schwefel2_13(self.rotation_non_identity, self.shift, 2)
        schwefel.a = self.static_a
        schwefel.b = self.static_b
        schwefel.alpha = self.static_alpha
        input_vector = [2.0, 2.0]  # Example input
        result = schwefel.evaluate(input_vector)
        self.assertIsInstance(result, float)  # Check if the result is a float

    def test_evaluate_with_shift_without_rotation(self):
        schwefel = Schwefel2_13(self.rotation_identity, self.shift, 2)
        schwefel.a = self.static_a
        schwefel.b = self.static_b
        schwefel.alpha = self.static_alpha
        input_vector = [2.0, 2.0]  # Example input
        result = schwefel.evaluate(input_vector)
        self.assertIsInstance(result, float)  # Check if the result is a float

    def test_evaluate_without_shift_with_rotation(self):
        schwefel = Schwefel2_13(self.rotation_non_identity, self.no_shift, 2)
        schwefel.a = self.static_a
        schwefel.b = self.static_b
        schwefel.alpha = self.static_alpha
        input_vector = [2.0, 2.0]  # Example input
        result = schwefel.evaluate(input_vector)
        self.assertIsInstance(result, float)  # Check if the result is a float

    def test_evaluate_without_shift_without_rotation(self):
        schwefel = Schwefel2_13(self.rotation_identity, self.no_shift, 2)
        schwefel.a = self.static_a
        schwefel.b = self.static_b
        schwefel.alpha = self.static_alpha
        input_vector = [2.0, 2.0]  # Example input
        result = schwefel.evaluate(input_vector)
        self.assertIsInstance(result, float)  # Check if the result is a float

    def test_evaluate_with_zero_input(self):
        schwefel = Schwefel2_13(self.rotation_identity, self.no_shift, 2)
        schwefel.a = self.static_a
        schwefel.b = self.static_b
        schwefel.alpha = self.static_alpha
        input_vector = [0.0, 0.0]  # Known input
        result = schwefel.evaluate(input_vector)
        expected_result = 0.0  # Expected result based on static values
        # Check if the result is as expected
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_negative_input(self):
        schwefel = Schwefel2_13(self.rotation_identity, self.shift, 2)
        schwefel.a = self.static_a
        schwefel.b = self.static_b
        schwefel.alpha = self.static_alpha
        input_vector = [-1.0, -1.0]  # Known input
        result = schwefel.evaluate(input_vector)
        self.assertIsInstance(result, float)  # Check if the result is a float

    def test_evaluate_with_large_input(self):
        schwefel = Schwefel2_13(self.rotation_identity, self.shift, 2)
        schwefel.a = self.static_a
        schwefel.b = self.static_b
        schwefel.alpha = self.static_alpha
        input_vector = [100.0, 100.0]  # Known input
        result = schwefel.evaluate(input_vector)
        self.assertIsInstance(result, float)  # Check if the result is a float

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
