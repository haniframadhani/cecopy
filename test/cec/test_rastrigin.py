import unittest
import math
from ceco.cec.rastrigin import Rastrigin


class Test_ackley(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        rastrigin = Rastrigin(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]
        result = rastrigin.evaluate(input_vector)
        expected_result = (2**2-10*math.cos(2*math.pi*2)+10) + \
            (3**2-10*math.cos(2*math.pi*3)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        rastrigin = Rastrigin(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]
        result = rastrigin.evaluate(input_vector)
        expected_result = ((-3)**2-10*math.cos(2*math.pi*(-3))+10) + \
            (2**2-10*math.cos(2*math.pi*2)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        rastrigin = Rastrigin(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = rastrigin.evaluate(input_vector)
        expected_result = (2**2-10*math.cos(2*math.pi*2)+10) + \
            ((-1)**2-10*math.cos(2*math.pi*(-1))+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        rastrigin = Rastrigin(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = rastrigin.evaluate(input_vector)
        expected_result = (1**2-10*math.cos(2*math.pi*1)+10) + \
            (2**2-10*math.cos(2*math.pi*2)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        rastrigin = Rastrigin(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]
        result = rastrigin.evaluate(input_vector)
        expected_result = (0**2-10*math.cos(2*math.pi*0)+10) + \
            (0**2-10*math.cos(2*math.pi*0)+10)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Rastrigin(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Rastrigin(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Rastrigin(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Rastrigin(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        rastrigin = Rastrigin(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            rastrigin.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
