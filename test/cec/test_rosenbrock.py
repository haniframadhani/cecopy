import unittest
from ceco.cec.rosenbrock import Rosenbrock


class Test_rosenbrock(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.f_bias = 1.0

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        rosenbrock = Rosenbrock(self.rotation_identity, self.no_shift)
        input_vector = [1.0, 1.0]  # Example input
        result = rosenbrock.evaluate(input_vector)
        expected_result = 0.0  # Minimum value of the Rosenbrock function
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        rosenbrock = Rosenbrock(self.rotation_identity, self.shift)
        input_vector = [1.0, 1.0]  # Example input
        result = rosenbrock.evaluate(input_vector)
        # (0^2) = 0, shifted gives (1^2) = 1
        expected_result = 1.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        rosenbrock = Rosenbrock(self.rotation_non_identity, self.no_shift)
        input_vector = [1.0, 1.0]  # Example input
        result = rosenbrock.evaluate(input_vector)
        # After rotation: [1.0, -1.0] remains the same
        expected_result = 400.0  # Minimum value of the Rosenbrock function
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        rosenbrock = Rosenbrock(self.rotation_non_identity, self.shift)
        input_vector = [1.0, 1.0]  # Example input
        result = rosenbrock.evaluate(input_vector)
        expected_result = 1.0  # Minimum value of the Rosenbrock function
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        rosenbrock = Rosenbrock(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]  # Known input
        result = rosenbrock.evaluate(input_vector)
        expected_result = 100 * (0.0**2 - 0)**2 + (0.0 - 1)**2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        rosenbrock = Rosenbrock(self.rotation_identity,
                                self.no_shift, self.f_bias)
        input_vector = [0.0, 0.0]  # Known input
        result = rosenbrock.evaluate(input_vector)
        expected_result = 100 * (0.0**2 - 0)**2 + (0.0 - 1)**2 + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Rosenbrock(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Rosenbrock(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Rosenbrock(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Rosenbrock(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        rosenbrock = Rosenbrock(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            rosenbrock.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
