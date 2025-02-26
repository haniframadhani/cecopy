import unittest
import random
from ceco.bbob import Bbob
from ceco.coco.rastrigin import Rastrigin
import math


class Test_rastrigin(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_initialization(self):
        dimension = 5
        rastrigin = Rastrigin(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(rastrigin.x_opt), dimension)
        for x in rastrigin.x_opt:
            self.assertTrue(-5 <= x <= 5)

        # Check if f_opt is computed correctly
        expected_f_opt = rastrigin.raw(rastrigin.x_opt)
        self.assertEqual(rastrigin.f_opt, expected_f_opt)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = [1.3942679845788373, -4.74989244777333, -
                          2.2497068163088074, -2.7678926185117723, 2.3647121416401244]
        self.assertEqual(rastrigin.x_opt, expected_x_opt)

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        rastrigin = Rastrigin(dimension)

        # Evaluate at x_opt
        result = rastrigin.evaluate(rastrigin.x_opt)
        self.assertAlmostEqual(result, rastrigin.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        rastrigin = Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Zero vector
        input_vector = [0.0] * dimension
        result = rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, rastrigin.x_opt)]
        z = bbob.T_osz(z)
        z = bbob.matrix_multiply(
            bbob.create_diagonal_matrix(10), bbob.T_asy_beta(0.2, z))
        sum_cos = sum(math.cos(2 * math.pi * x_i) for x_i in z)
        expected_result = 10 * (dimension - sum_cos)
        expected_result = expected_result + \
            bbob.euclidean_norm(z) + rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        rastrigin = Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = [-3, -2]
        result = rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, rastrigin.x_opt)]
        z = bbob.T_osz(z)
        z = bbob.matrix_multiply(
            bbob.create_diagonal_matrix(10), bbob.T_asy_beta(0.2, z))
        sum_cos = sum(math.cos(2 * math.pi * x_i) for x_i in z)
        expected_result = 10 * (dimension - sum_cos)
        expected_result = expected_result + \
            bbob.euclidean_norm(z) + rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_custom_f_opt(self):
        dimension = 3
        custom_f_opt = 10.0
        rastrigin = Rastrigin(dimension, f_opt=custom_f_opt)

        # Evaluate at x_opt
        result = rastrigin.evaluate(rastrigin.x_opt)
        self.assertAlmostEqual(result, custom_f_opt, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        rastrigin = Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Evaluate at x_opt
        result = rastrigin.evaluate(rastrigin.x_opt)
        self.assertAlmostEqual(result, rastrigin.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = [2.0]
        result = rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, rastrigin.x_opt)]
        z = bbob.T_osz(z)
        z = bbob.matrix_multiply(
            bbob.create_diagonal_matrix(10), bbob.T_asy_beta(0.2, z))
        sum_cos = sum(math.cos(2 * math.pi * x_i) for x_i in z)
        expected_result = 10 * (dimension - sum_cos)
        expected_result = expected_result + \
            bbob.euclidean_norm(z) + rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        rastrigin = Rastrigin(dimension)

        # Empty input vector
        input_vector = []
        with self.assertRaises(ValueError):
            rastrigin.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
