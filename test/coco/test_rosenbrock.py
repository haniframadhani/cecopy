import unittest
from ceco.coco.rosenbrock import Rosenbrock
from ceco.benchmark import Benchmark
import numpy as np


class Test_rosenbrock(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        rotation = np.eye(dimension)
        test_func = Rosenbrock(dimension, rotation)

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
        rotation = np.eye(dimension)
        test_func = Rosenbrock(dimension, rotation)

        # Evaluate at x_opt
        result = test_func.evaluate(test_func.x_opt)

        # Manually compute the expected result
        scaling_factor = max(1, test_func.dimension / 8)
        z = scaling_factor * (test_func.x_opt - test_func.x_opt) + 1
        expected_result = test_func.raw(z) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_at_zero_vector(self):
        dimension = 3
        rotation = np.eye(dimension)
        test_func = Rosenbrock(dimension, rotation)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, test_func.dimension / 8)
        z = scaling_factor * (input_vector - test_func.x_opt) + 1
        expected_result = test_func.raw(z) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        rotation = np.eye(dimension)
        test_func = Rosenbrock(dimension, rotation)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, test_func.dimension / 8)
        z = scaling_factor * (input_vector - test_func.x_opt) + 1
        expected_result = test_func.raw(z) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        rotation = np.eye(dimension)
        test_func = Rosenbrock(dimension, rotation)

        # Evaluate at x_opt
        result = test_func.evaluate(test_func.x_opt)
        self.assertAlmostEqual(result, test_func.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, test_func.dimension / 8)
        z = scaling_factor * (input_vector - test_func.x_opt) + 1
        expected_result = test_func.raw(z) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        rotation = np.eye(dimension)
        test_func = Rosenbrock(dimension, rotation)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            test_func.evaluate(input_vector)

    def test_initialization_rotated(self):
        dimension = 5
        rotation = np.array([[0, -1, 0, 0, 0],
                            [1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]])
        test_func = Rosenbrock(dimension, rotation)

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

    # def test_evaluate_at_optimal_point_rotated(self):
    #     dimension = 3
    #     rotation = np.array([[0, -1, 0],
    #                         [1, 0, 0],
    #                         [0, 0, 1]])
    #     test_func = Rosenbrock(dimension, rotation)

    #     # Evaluate at x_opt
    #     result = test_func.evaluate(test_func.x_opt)
    #     self.assertAlmostEqual(result, test_func.f_opt, places=6)

    def test_evaluate_at_zero_vector_rotated(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 3
        rotation = np.array([[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])
        test_func = Rosenbrock(dimension, rotation)
        benchmark = Benchmark(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, test_func.dimension / 8)
        z = scaling_factor * \
            np.matmul(benchmark.gram_schmidt(rotation.T), input_vector) + 1 / 2
        expected_result = test_func.raw(z) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values_rotated(self):
        dimension = 3
        rotation = np.array([[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])
        test_func = Rosenbrock(dimension, rotation)
        benchmark = Benchmark(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2, -1])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, test_func.dimension / 8)
        z = scaling_factor * \
            np.matmul(benchmark.gram_schmidt(rotation.T), input_vector) + 1 / 2
        expected_result = test_func.raw(z) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1_rotated(self):
        dimension = 1
        rotation = np.array([[-1]])
        test_func = Rosenbrock(dimension, rotation)
        benchmark = Benchmark(dimension)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, test_func.dimension / 8)
        z = scaling_factor * \
            np.matmul(benchmark.gram_schmidt(rotation.T), input_vector) + 1 / 2
        expected_result = test_func.raw(z) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)


if __name__ == '__main__':
    unittest.main()
