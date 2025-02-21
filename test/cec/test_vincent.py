import unittest
from ceco.cec.vincent import Vincent


class Test_vincent(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.f_bias = 1.0

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        vincent = Vincent(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = vincent.evaluate(input_vector)
        expected_result = -0.1980669543
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        vincent = Vincent(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        shifted_vector = [1.0, 2.0]
        result = vincent.evaluate(input_vector)
        expected_result = 0.3019107135584
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        vincent = Vincent(self.rotation_non_identity, self.no_shift)
        input_vector = [-2.0, 3.0]  # Example input
        rotated_vector = [3.0, 2.0]
        result = vincent.evaluate(input_vector)
        expected_result = -0.19806695436314
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        vincent = Vincent(self.rotation_non_identity, self.shift)
        input_vector = [-2.0, 3.0]  # Example input
        shifted_rotated_vector = [2.0, 3.0]
        result = vincent.evaluate(input_vector)
        expected_result = -0.1980669543
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        vincent = Vincent(self.rotation_identity, self.no_shift,
                          self.f_bias)
        input_vector = [2.0, 3.0]  # Known input
        result = vincent.evaluate(input_vector)
        expected_result = (-0.1980669543) + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Vincent(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Vincent(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Vincent(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Vincent(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        vincent = Vincent(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            vincent.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")

    def test_evaluate_with_input_less_or_equal_than_zero(self):
        vincent = Vincent(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        rotated_vector = [3.0, -2.0]
        with self.assertRaises(ValueError) as context:
            vincent.evaluate(input_vector)
        self.assertEqual(str(context.exception),
                         "All elements in input_vector must be greater than 0 after shift and rotation")


if __name__ == '__main__':
    unittest.main()
