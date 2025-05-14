import unittest
from ceco.cec.expanded_schaffer_f6 import Expanded_schaffer_f6
import numpy as np


class Test_schaffer(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = np.eye(2)
        self.rotation_non_identity = np.array([[0, -1], [1, 0]])
        self.shift = np.array([1, 1])
        self.no_shift = np.zeros(2)
        self.f_bias = 1.0
        self.dimension = 2

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        schaffer = Expanded_schaffer_f6(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = schaffer.evaluate(input_vector)
        expected_result = 0.0
        for i in range(2 - 1):
            expected_result += schaffer.schaffer_base(
                input_vector[i], input_vector[i + 1])

        expected_result += schaffer.schaffer_base(
            input_vector[-1], input_vector[0])
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        schaffer = Expanded_schaffer_f6(
            self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        shifted_vector = np.array([2.0, 1.0])
        result = schaffer.evaluate(input_vector)
        expected_result = 0.0
        for i in range(2 - 1):
            expected_result += schaffer.schaffer_base(
                shifted_vector[i], shifted_vector[i + 1])

        expected_result += schaffer.schaffer_base(
            shifted_vector[-1], shifted_vector[0])
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        schaffer = Expanded_schaffer_f6(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        rotated_vector = np.array([-2.0, 3.0])
        result = schaffer.evaluate(input_vector)
        expected_result = 0.0
        for i in range(2 - 1):
            expected_result += schaffer.schaffer_base(
                rotated_vector[i], rotated_vector[i + 1])

        expected_result += schaffer.schaffer_base(
            rotated_vector[-1], rotated_vector[0])
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        schaffer = Expanded_schaffer_f6(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        z = np.array([1.0, -2.0])
        result = schaffer.evaluate(input_vector)
        expected_result = 0.0
        for i in range(2 - 1):
            expected_result += schaffer.schaffer_base(
                z[i], z[i + 1])

        expected_result += schaffer.schaffer_base(
            z[-1], z[0])
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        schaffer = Expanded_schaffer_f6(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = schaffer.evaluate(input_vector)
        expected_result = 0.0
        for i in range(2 - 1):
            expected_result += schaffer.schaffer_base(
                input_vector[i], input_vector[i + 1])

        expected_result += schaffer.schaffer_base(
            input_vector[-1], input_vector[0])
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        schaffer = Expanded_schaffer_f6(
            self.dimension, self.rotation_identity, self.no_shift, self.f_bias)
        input_vector = np.zeros(2)
        result = schaffer.evaluate(input_vector)
        expected_result = 0.0
        for i in range(2 - 1):
            expected_result += schaffer.schaffer_base(
                input_vector[i], input_vector[i + 1])

        expected_result += schaffer.schaffer_base(
            input_vector[-1], input_vector[0])
        expected_result += self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
