import unittest
import numpy as np
from ceco.benchmark import Benchmark
from ceco.coco.buche_rastrigin import Buche_rastrigin


class TestBucheRastrigin(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        test_func = Buche_rastrigin(dimension)

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
        test_func = Buche_rastrigin(dimension)
        benchmark = Benchmark(dimension)

        result = test_func.evaluate(test_func.x_opt)

        # Manually compute expected result
        z = test_func.x_opt - test_func.x_opt
        z = benchmark.T_osz(z)
        s = test_func.compute_s_i(z)
        z = benchmark.elementwise_multiply(z, s)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        sum_square = np.sum(z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            benchmark.f_pen(test_func.x_opt) + test_func.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_at_zero_vector(self):
        dimension = 2
        test_func = Buche_rastrigin(dimension)
        benchmark = Benchmark(dimension)

        input_vector = np.zeros(dimension)
        result = test_func.evaluate(input_vector)

        # Manually compute expected result
        z = input_vector - test_func.x_opt
        z = benchmark.T_osz(z)
        s = test_func.compute_s_i(z)
        z = benchmark.elementwise_multiply(z, s)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        sum_square = np.sum(z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            benchmark.f_pen(input_vector) + test_func.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        test_func = Buche_rastrigin(dimension)
        benchmark = Benchmark(dimension)

        input_vector = np.array([-3, -2])
        result = test_func.evaluate(input_vector)

        z = input_vector - test_func.x_opt
        z = benchmark.T_osz(z)
        s = test_func.compute_s_i(z)
        z = benchmark.elementwise_multiply(z, s)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        sum_square = np.sum(z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            benchmark.f_pen(input_vector) + test_func.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        test_func = Buche_rastrigin(dimension)
        benchmark = Benchmark(dimension)

        input_vector = np.array([2.0])
        result = test_func.evaluate(input_vector)

        z = input_vector - test_func.x_opt
        z = benchmark.T_osz(z)
        s = test_func.compute_s_i(z)
        z = benchmark.elementwise_multiply(z, s)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        sum_square = np.sum(z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            benchmark.f_pen(input_vector) + test_func.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        test_func = Buche_rastrigin(dimension)

        input_vector = np.array([])
        with self.assertRaises(ValueError):
            test_func.evaluate(input_vector)

    # Test Compute s_i
    def test_compute_s_i(self):
        dimension = 3
        test_func = Buche_rastrigin(dimension)
        z_i = np.array([1, 2, 3])
        result = test_func.compute_s_i(z_i)

        expected = np.array([
            10 * (10 ** (0.5 * (0) / 2)),
            10 ** (0.5 * (1) / 2),
            10 * (10 ** (0.5 * (2) / 2))
        ])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


if __name__ == '__main__':
    unittest.main()
