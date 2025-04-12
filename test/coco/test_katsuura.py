import unittest
from ceco.benchmark import Benchmark
from ceco.coco.katsuura import Katsuura
import numpy as np


class Test_katsuura(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        test_func = Katsuura(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(test_func.x_opt), dimension)
        self.assertTrue(np.all(test_func.x_opt >= -5)
                        and np.all(test_func.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = test_func.raw(test_func.x_opt)
        self.assertAlmostEqual(test_func.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(test_func.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        test_func = Katsuura(dimension)

        # Evaluate at x_opt
        result = test_func.evaluate(test_func.x_opt)
        self.assertAlmostEqual(result, test_func.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        test_func = Katsuura(dimension)
        benchmark = Benchmark(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - test_func.x_opt
        z = test_func.Q @ test_func.diagonal_matrix @ test_func.R @ z
        j_values = np.arange(1, 33)
        powers_of_two = 2 ** j_values
        exponent = 10 / (dimension ** 1.2)
        scale_factor = 10 / (dimension ** 2)
        powers_of_2 = 2 ** np.arange(1, 33)

        # Compute sum term using vectorized operations
        sum_terms = np.sum(np.abs(powers_of_2[:, None] * z - np.round(
            powers_of_2[:, None] * z)) / powers_of_2[:, None], axis=0)

        # Compute product term efficiently
        indices = np.arange(1, dimension + 1)
        product = np.prod((1 + indices * sum_terms) **
                          (10 / (dimension ** 1.2)))

        # Compute final result
        expected_result = (10 / (dimension ** 2)) * product - \
            (10 / (dimension ** 2)) + \
            benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        test_func = Katsuura(dimension)
        benchmark = Benchmark(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - test_func.x_opt
        z = test_func.Q @ test_func.diagonal_matrix @ test_func.R @ z
        j_values = np.arange(1, 33)
        powers_of_two = 2 ** j_values
        exponent = 10 / (dimension ** 1.2)
        scale_factor = 10 / (dimension ** 2)
        powers_of_2 = 2 ** np.arange(1, 33)

        # Compute sum term using vectorized operations
        sum_terms = np.sum(np.abs(powers_of_2[:, None] * z - np.round(
            powers_of_2[:, None] * z)) / powers_of_2[:, None], axis=0)

        # Compute product term efficiently
        indices = np.arange(1, dimension + 1)
        product = np.prod((1 + indices * sum_terms) **
                          (10 / (dimension ** 1.2)))

        # Compute final result
        expected_result = (10 / (dimension ** 2)) * product - \
            (10 / (dimension ** 2)) + \
            benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        test_func = Katsuura(dimension)
        benchmark = Benchmark(dimension)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - test_func.x_opt
        z = test_func.Q @ test_func.diagonal_matrix @ test_func.R @ z
        j_values = np.arange(1, 33)
        powers_of_two = 2 ** j_values
        exponent = 10 / (dimension ** 1.2)
        scale_factor = 10 / (dimension ** 2)
        powers_of_2 = 2 ** np.arange(1, 33)

        # Compute sum term using vectorized operations
        sum_terms = np.sum(np.abs(powers_of_2[:, None] * z - np.round(
            powers_of_2[:, None] * z)) / powers_of_2[:, None], axis=0)

        # Compute product term efficiently
        indices = np.arange(1, dimension + 1)
        product = np.prod((1 + indices * sum_terms) **
                          (10 / (dimension ** 1.2)))

        # Compute final result
        expected_result = (10 / (dimension ** 2)) * product - \
            (10 / (dimension ** 2)) + \
            benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        test_func = Katsuura(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            test_func.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
