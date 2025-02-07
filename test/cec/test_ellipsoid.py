import unittest
from ceco.cec.ellipsoid import Ellipsoid


class Test_ellipsoid(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        elliptic = Ellipsoid(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        expected_result = 1 * (2.0 ** 2) + 2 * \
            (3.0 ** 2)  # F(X) = 1*(2^2) + 2*(3^2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        elliptic = Ellipsoid(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        expected_result = 1 * (2.0 - 1) ** 2 + 2 * \
            (3.0 - 1) ** 2  # F(X) = 1*(1^2) + 2*(2^2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        elliptic = Ellipsoid(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        # After rotation: [3.0, 2.0]
        expected_result = 1 * (3.0 ** 2) + 2 * \
            (2.0 ** 2)  # F(X) = 1*(3^2) + 2*(2^2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        elliptic = Ellipsoid(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        # After rotation: [3.0, 2.0]
        # After shift: [-3.0 - 1, 2.0 - 1] = [-4.0, 1.0]
        expected_result = (1 * ((-4.0) ** 2)) + \
            (2 * (1.0 ** 2))  # F(X) = 1*(-4^2) + 2*(1^2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        elliptic = Ellipsoid(self.rotation_identity, self.shift)
        input_vector = [0.0, 0.0]  # Known input
        result = elliptic.evaluate(input_vector)
        expected_result = 1 * (0.0 - 1) ** 2 + 2 * \
            (0.0 - 1) ** 2  # F(X) = 1*(1^2) + 2*(1^2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Ellipsoid(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Ellipsoid(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Ellipsoid(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Ellipsoid(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        ellipsoid = Ellipsoid(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            ellipsoid.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
