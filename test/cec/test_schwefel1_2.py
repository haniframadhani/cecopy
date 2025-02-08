import unittest
from ceco.cec.schwefel1_2 import Schwefel1_2


class Test_schwefel1_2(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        schwefel1_2 = Schwefel1_2(self.rotation_identity, self.no_shift)
        input_vector = [3.0, 2.0]  # Example input
        result = schwefel1_2.evaluate(input_vector)
        expected_result = 34.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        schwefel1_2 = Schwefel1_2(self.rotation_identity, self.shift)
        input_vector = [3.0, 2.0]  # Example input
        result = schwefel1_2.evaluate(input_vector)
        expected_result = 13.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        schwefel1_2 = Schwefel1_2(self.rotation_non_identity, self.no_shift)
        input_vector = [3.0, 2.0]  # Example input
        result = schwefel1_2.evaluate(input_vector)
        expected_result = 5.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        schwefel1_2 = Schwefel1_2(self.rotation_non_identity, self.shift)
        input_vector = [3.0, 2.0]  # Example input
        result = schwefel1_2.evaluate(input_vector)
        expected_result = 2.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        schwefel1_2 = Schwefel1_2(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]  # Known input
        result = schwefel1_2.evaluate(input_vector)
        expected_result = 0.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Schwefel1_2(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Schwefel1_2(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Schwefel1_2(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Schwefel1_2(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        schwefel1_2 = Schwefel1_2(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            schwefel1_2.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
