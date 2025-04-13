import unittest
import numpy as np
from ceco.benchmark import Benchmark
from ceco.coco.schaffer_f7 import Schaffer_f7


class Test_schaffer_f7(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        test_func = Schaffer_f7(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(test_func.x_opt), dimension)
        self.assertTrue(np.all(test_func.x_opt >= 0)
                        and np.all(test_func.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = test_func.raw(test_func.x_opt)
        self.assertAlmostEqual(test_func.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        print(test_func.x_opt)
        expected_x_opt = np.array(
            [1.87270059, 4.75357153, 3.65996971, 2.99329242, 0.7800932])
        self.assertTrue(np.allclose(test_func.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        test_func = Schaffer_f7(dimension)

        # Evaluate at x_opt
        result = test_func.evaluate(test_func.x_opt)
        self.assertAlmostEqual(result, test_func.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        test_func = Schaffer_f7(dimension)
        benchmark = Benchmark(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z = test_func.diagonal_matrix @ test_func.Q @ benchmark.T_asy_beta(
            0.5, test_func.R @ (input_vector - test_func.x_opt))
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        sqrt_s = np.sqrt(s)
        s_pow = np.power(s, 0.2)
        sin_term = np.sin(50 * s_pow)
        inner_term = sqrt_s + sqrt_s * np.square(sin_term)
        total_sum = np.sum(inner_term)
        expected_result = ((total_sum / (dimension - 1)
                            ) ** 2) + 10 * benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        test_func = Schaffer_f7(dimension)
        benchmark = Benchmark(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z = test_func.diagonal_matrix @ test_func.Q @ benchmark.T_asy_beta(
            0.5, test_func.R @ (input_vector - test_func.x_opt))
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        sqrt_s = np.sqrt(s)
        s_pow = np.power(s, 0.2)
        sin_term = np.sin(50 * s_pow)
        inner_term = sqrt_s + sqrt_s * np.square(sin_term)
        total_sum = np.sum(inner_term)
        expected_result = ((total_sum / (dimension - 1)
                            ) ** 2) + 10 * benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        with self.assertRaises(ValueError):
            Schaffer_f7(dimension)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        test_func = Schaffer_f7(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            test_func.evaluate(input_vector)

    def test_initialization_ill(self):
        dimension = 5
        test_func = Schaffer_f7(dimension, ill_conditioned=True)

        # Check if x_opt is generated correctly
        self.assertEqual(len(test_func.x_opt), dimension)
        self.assertTrue(np.all(test_func.x_opt >= -5)
                        and np.all(test_func.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = test_func.raw(test_func.x_opt)
        self.assertAlmostEqual(test_func.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [1.87270059, 4.75357153, 3.65996971, 2.99329242, 0.7800932])
        self.assertTrue(np.allclose(test_func.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    def test_evaluate_at_optimal_point_ill(self):
        dimension = 3
        test_func = Schaffer_f7(dimension, ill_conditioned=True)

        # Evaluate at x_opt
        result = test_func.evaluate(test_func.x_opt)
        self.assertAlmostEqual(result, test_func.f_opt, places=6)

    def test_evaluate_at_zero_vector_ill(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        test_func = Schaffer_f7(dimension, ill_conditioned=True)
        benchmark = Benchmark(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z = test_func.diagonal_matrix @ test_func.Q @ benchmark.T_asy_beta(
            0.5, test_func.R @ (input_vector - test_func.x_opt))
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        sqrt_s = np.sqrt(s)
        s_pow = np.power(s, 0.2)
        sin_term = np.sin(50 * s_pow)
        inner_term = sqrt_s + sqrt_s * np.square(sin_term)
        total_sum = np.sum(inner_term)
        expected_result = ((total_sum / (dimension - 1)
                            ) ** 2) + 10 * benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values_ill(self):
        dimension = 2
        test_func = Schaffer_f7(dimension, ill_conditioned=True)
        benchmark = Benchmark(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z = test_func.diagonal_matrix @ test_func.Q @ benchmark.T_asy_beta(
            0.5, test_func.R @ (input_vector - test_func.x_opt))
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        sqrt_s = np.sqrt(s)
        s_pow = np.power(s, 0.2)
        sin_term = np.sin(50 * s_pow)
        inner_term = sqrt_s + sqrt_s * np.square(sin_term)
        total_sum = np.sum(inner_term)
        expected_result = ((total_sum / (dimension - 1)
                            ) ** 2) + 10 * benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1_ill(self):
        dimension = 1
        with self.assertRaises(ValueError):
            Schaffer_f7(dimension, ill_conditioned=True)

    def test_evaluate_with_empty_input_vector_ill(self):
        dimension = 3
        test_func = Schaffer_f7(dimension, ill_conditioned=True)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            test_func.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
