import unittest
import numpy as np
from ceco.benchmark import Benchmark
from ceco.coco.step_ellipsoidal import Step_ellipsoidal


class Test_ellipsoidal(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        test_func = Step_ellipsoidal(dimension)

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
        test_func = Step_ellipsoidal(dimension)
        benchmark = Benchmark(dimension)

        # Evaluate at x_opt
        result = test_func.evaluate(test_func.x_opt)

        # Manually compute the expected result
        z_hat = np.matmul(benchmark.create_diagonal_matrix(
            10), np.matmul(test_func.R, test_func.x_opt - test_func.x_opt))

        z_tilde = np.zeros_like(z_hat)
        for i in range(z_hat.shape[0]):
            if np.abs(z_hat[i]) > 0.5:
                z_tilde[i] = np.floor(0.5+z_hat[i])
            else:
                z_tilde[i] = np.floor(0.5+10*z_hat[i])/10

        z = np.matmul(test_func.Q, z_tilde)

        term1 = np.abs(z_hat[0]) / 1e4

        exponents = 2 * (np.arange(dimension)-1)/(dimension - 1)
        weight = 10 ** exponents
        term2 = np.sum(weight * z ** 2)

        expected_result = 0.1 * max(term1, term2) + \
            benchmark.f_pen(test_func.x_opt) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        test_func = Step_ellipsoidal(dimension)
        benchmark = Benchmark(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z_hat = np.matmul(benchmark.create_diagonal_matrix(
            10), np.matmul(test_func.R, input_vector - test_func.x_opt))

        z_tilde = np.zeros_like(z_hat)
        for i in range(z_hat.shape[0]):
            if np.abs(z_hat[i]) > 0.5:
                z_tilde[i] = np.floor(0.5+z_hat[i])
            else:
                z_tilde[i] = np.floor(0.5+10*z_hat[i])/10

        z = np.matmul(test_func.Q, z_tilde)

        term1 = np.abs(z_hat[0]) / 1e4

        exponents = 2 * (np.arange(dimension)-1)/(dimension - 1)
        weight = 10 ** exponents
        term2 = np.sum(weight * z ** 2)

        expected_result = 0.1 * max(term1, term2) + \
            benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        test_func = Step_ellipsoidal(dimension)
        benchmark = Benchmark(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z_hat = np.matmul(benchmark.create_diagonal_matrix(
            10), np.matmul(test_func.R, input_vector - test_func.x_opt))

        z_tilde = np.zeros_like(z_hat)
        for i in range(z_hat.shape[0]):
            if np.abs(z_hat[i]) > 0.5:
                z_tilde[i] = np.floor(0.5+z_hat[i])
            else:
                z_tilde[i] = np.floor(0.5+10*z_hat[i])/10

        z = np.matmul(test_func.Q, z_tilde)

        term1 = np.abs(z_hat[0]) / 1e4

        exponents = 2 * (np.arange(dimension)-1)/(dimension - 1)
        weight = 10 ** exponents
        term2 = np.sum(weight * z ** 2)

        expected_result = 0.1 * max(term1, term2) + \
            benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        test_func = Step_ellipsoidal(dimension)
        benchmark = Benchmark(dimension)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = test_func.evaluate(input_vector)

        # Manually compute the expected result
        z_hat = np.matmul(benchmark.create_diagonal_matrix(
            10), np.matmul(test_func.R, input_vector - test_func.x_opt))

        z_tilde = np.zeros_like(z_hat)
        for i in range(z_hat.shape[0]):
            if np.abs(z_hat[i]) > 0.5:
                z_tilde[i] = np.floor(0.5+z_hat[i])
            else:
                z_tilde[i] = np.floor(0.5+10*z_hat[i])/10

        z = np.matmul(test_func.Q, z_tilde)

        term1 = np.abs(z_hat[0]) / 1e4

        exponents = 0
        weight = 10 ** exponents
        term2 = np.sum(weight * z ** 2)

        expected_result = 0.1 * max(term1, term2) + \
            benchmark.f_pen(input_vector) + test_func.f_opt
        expected_result = 0.1 * max(term1, term2) + \
            benchmark.f_pen(input_vector) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        test_func = Step_ellipsoidal(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            test_func.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
