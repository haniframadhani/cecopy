import numpy as np
import unittest
from ceco.cec.katsuura import Katsuura


class Test_katsuura(unittest.TestCase):
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
        katsuura = Katsuura(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = katsuura.evaluate(input_vector)

        product = 1.0
        for i in range(1, 2 + 1):
            sum_term = 0.0
            for j in range(1, 33):
                term = np.abs(
                    (2**j)*input_vector[i-1] - np.round((2**j)*input_vector[i-1]))/(2**j)
                sum_term += term
            product *= (1 + i * sum_term)**(10 / (2**1.2))

        expected_result = (10 / (2**2)) * product - \
            (10/(2**2))
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        katsuura = Katsuura(self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        shifted_vector = np.array([1.0, 2.0])
        result = katsuura.evaluate(input_vector)

        product = 1.0
        for i in range(1, 2 + 1):
            sum_term = 0.0
            for j in range(1, 33):
                term = np.abs(
                    (2**j)*shifted_vector[i-1] - np.round((2**j)*shifted_vector[i-1]))/(2**j)
                sum_term += term
            product *= (1 + i * sum_term)**(10 / (2**1.2))

        expected_result = (10 / (2**2)) * product - \
            (10/(2**2))
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        katsuura = Katsuura(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        rotated_vector = np.array([-3.0, 2.0])
        result = katsuura.evaluate(input_vector)

        product = 1.0
        for i in range(1, 2 + 1):
            sum_term = 0.0
            for j in range(1, 33):
                term = np.abs(
                    (2**j)*rotated_vector[i-1] - np.round((2**j)*rotated_vector[i-1]))/(2**j)
                sum_term += term
            product *= (1 + i * sum_term)**(10 / (2**1.2))

        expected_result = (10 / (2**2)) * product - \
            (10/(2**2))
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        katsuura = Katsuura(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        shifted_rotated_vector = np.array([-2.0, 1.0])
        result = katsuura.evaluate(input_vector)

        product = 1.0
        for i in range(1, 2 + 1):
            sum_term = 0.0
            for j in range(1, 33):
                term = np.abs(
                    (2**j)*shifted_rotated_vector[i-1] - np.round((2**j)*shifted_rotated_vector[i-1]))/(2**j)
                sum_term += term
            product *= (1 + i * sum_term)**(10 / (2**1.2))

        expected_result = (10 / (2**2)) * product - \
            (10/(2**2))
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        katsuura = Katsuura(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.zeros(2)
        result = katsuura.evaluate(input_vector)

        product = 1.0
        for i in range(1, 2 + 1):
            sum_term = 0.0
            for j in range(1, 33):
                term = np.abs(
                    (2**j)*input_vector[i-1] - np.round((2**j)*input_vector[i-1]))/(2**j)
                sum_term += term
            product *= (1 + i * sum_term)**(10 / (2**1.2))

        expected_result = (10 / (2**2)) * product - \
            (10/(2**2))
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        katsuura = Katsuura(self.dimension, self.rotation_identity, self.no_shift,
                            self.f_bias)
        input_vector = np.zeros(2)
        result = katsuura.evaluate(input_vector)

        product = 1.0
        for i in range(1, 2 + 1):
            sum_term = 0.0
            for j in range(1, 33):
                term = np.abs(
                    (2**j)*input_vector[i-1] - np.round((2**j)*input_vector[i-1]))/(2**j)
                sum_term += term
            product *= (1 + i * sum_term)**(10 / (2**1.2))

        expected_result = (10 / (2**2)) * product - \
            (10/(2**2)) + self.f_bias
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
