import unittest
import math
from ceco.cec.weierstrass import Weierstrass


class Test_weierstrass(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector
        self.a = 0.5
        self.b = 3
        self.k_max = 20

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        weierstrass = Weierstrass(self.rotation_identity, self.no_shift)
        input_vector = [1.0, 50.0]
        result = weierstrass.evaluate(input_vector)
        total_sum = 0.0
        for i in range(2):
            inner_sum = 0.0
            for k in range(self.k_max + 1):
                inner_sum += self.a ** k * \
                    math.cos(2 * math.pi * self.b **
                             k * (input_vector[i] + 0.5))
            total_sum += inner_sum
        second_sum = 0.0
        for k in range(self.k_max + 1):
            second_sum += self.a ** k * \
                math.cos(2 * math.pi * self.b ** k * 0.5)
        expected_result = total_sum - 2 * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        weierstrass = Weierstrass(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = weierstrass.evaluate(input_vector)
        shift_vector = [1.0, 2.0]
        total_sum = 0.0
        for i in range(2):
            inner_sum = 0.0
            for k in range(self.k_max + 1):
                inner_sum += self.a ** k * \
                    math.cos(2 * math.pi * self.b **
                             k * (shift_vector[i] + 0.5))
            total_sum += inner_sum
        second_sum = 0.0
        for k in range(self.k_max + 1):
            second_sum += self.a ** k * \
                math.cos(2 * math.pi * self.b ** k * 0.5)
        expected_result = total_sum - 2 * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        weierstrass = Weierstrass(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = weierstrass.evaluate(input_vector)
        z = [-4.0, 1.0]
        total_sum = 0.0
        for i in range(2):
            inner_sum = 0.0
            for k in range(self.k_max + 1):
                inner_sum += self.a ** k * \
                    math.cos(2 * math.pi * self.b **
                             k * (z[i] + 0.5))
            total_sum += inner_sum
        second_sum = 0.0
        for k in range(self.k_max + 1):
            second_sum += self.a ** k * \
                math.cos(2 * math.pi * self.b ** k * 0.5)
        expected_result = total_sum - 2 * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        weierstrass = Weierstrass(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]
        result = weierstrass.evaluate(input_vector)
        rotated_vector = [-3.0, 2.0]
        total_sum = 0.0
        for i in range(2):
            inner_sum = 0.0
            for k in range(self.k_max + 1):
                inner_sum += self.a ** k * \
                    math.cos(2 * math.pi * self.b **
                             k * (rotated_vector[i] + 0.5))
            total_sum += inner_sum
        second_sum = 0.0
        for k in range(self.k_max + 1):
            second_sum += self.a ** k * \
                math.cos(2 * math.pi * self.b ** k * 0.5)
        expected_result = total_sum - 2 * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        weierstrass = Weierstrass(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]
        result = weierstrass.evaluate(input_vector)
        total_sum = 0.0
        for i in range(2):
            inner_sum = 0.0
            for k in range(self.k_max + 1):
                inner_sum += self.a ** k * \
                    math.cos(2 * math.pi * self.b **
                             k * (input_vector[i] + 0.5))
            total_sum += inner_sum
        second_sum = 0.0
        for k in range(self.k_max + 1):
            second_sum += self.a ** k * \
                math.cos(2 * math.pi * self.b ** k * 0.5)
        expected_result = total_sum - 2 * second_sum
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_rotation_not_list(self):
        with self.assertRaises(ValueError) as context:
            Weierstrass(rotation="invalid", shift=[0, 0])
            self.assertEqual(str(context.exception),
                             "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_empty(self):
        with self.assertRaises(ValueError) as context:
            Weierstrass(rotation=[], shift=[0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a non-empty list of lists")

    def test_invalid_rotation_not_square(self):
        with self.assertRaises(ValueError) as context:
            Weierstrass(rotation=[[1, 2, 3], [4, 5, 6]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "Rotation matrix must be a square matrix")

    def test_rotation_shift_mismatch(self):
        with self.assertRaises(ValueError) as context:
            Weierstrass(rotation=[[1, 0], [0, 1]], shift=[0, 0, 0])
        self.assertEqual(str(context.exception),
                         "rotation and shift has different dimensions")

    def test_input_vector_mismatch(self):
        weierstrass = Weierstrass(rotation=[[1, 0], [0, 1]], shift=[0, 0])
        with self.assertRaises(ValueError) as context:
            weierstrass.evaluate([1, 2, 3])  # Incorrect dimension
        self.assertEqual(str(context.exception),
                         "Input vector dimension does not match rotation and shift dimensions")


if __name__ == '__main__':
    unittest.main()
