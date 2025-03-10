import unittest
import numpy as np
from ceco.bbob import Bbob
from ceco.coco.buche_rastrigin import Buche_rastrigin


class TestBucheRastrigin(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        buche_rastrigin = Buche_rastrigin(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(buche_rastrigin.x_opt), dimension)
        self.assertTrue(np.all(buche_rastrigin.x_opt >= -5)
                        and np.all(buche_rastrigin.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = buche_rastrigin.raw(buche_rastrigin.x_opt)
        self.assertAlmostEqual(buche_rastrigin.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(buche_rastrigin.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        buche_rastrigin = Buche_rastrigin(dimension)

        result = buche_rastrigin.evaluate(buche_rastrigin.x_opt)
        self.assertAlmostEqual(result, buche_rastrigin.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        dimension = 2
        buche_rastrigin = Buche_rastrigin(dimension)
        bbob = Bbob(dimension)

        input_vector = np.zeros(dimension)
        result = buche_rastrigin.evaluate(input_vector)

        # Manually compute expected result
        z = input_vector - buche_rastrigin.x_opt
        z = bbob.T_osz(z)
        s = buche_rastrigin.compute_s_i(z)
        z = bbob.elementwise_multiply(z, s)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        sum_square = np.sum(z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            bbob.f_pen(input_vector) + buche_rastrigin.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        buche_rastrigin = Buche_rastrigin(dimension)
        bbob = Bbob(dimension)

        input_vector = np.array([-3, -2])
        result = buche_rastrigin.evaluate(input_vector)

        z = input_vector - buche_rastrigin.x_opt
        z = bbob.T_osz(z)
        s = buche_rastrigin.compute_s_i(z)
        z = bbob.elementwise_multiply(z, s)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        sum_square = np.sum(z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            bbob.f_pen(input_vector) + buche_rastrigin.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        buche_rastrigin = Buche_rastrigin(dimension)
        bbob = Bbob(dimension)

        input_vector = np.array([2.0])
        result = buche_rastrigin.evaluate(input_vector)

        z = input_vector - buche_rastrigin.x_opt
        z = bbob.T_osz(z)
        s = buche_rastrigin.compute_s_i(z)
        z = bbob.elementwise_multiply(z, s)
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        sum_square = np.sum(z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            bbob.f_pen(input_vector) + buche_rastrigin.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        buche_rastrigin = Buche_rastrigin(dimension)

        input_vector = np.array([])
        with self.assertRaises(ValueError):
            buche_rastrigin.evaluate(input_vector)

    # Test Compute s_i
    def test_compute_s_i(self):
        dimension = 3
        buche_rastrigin = Buche_rastrigin(dimension)
        z_i = np.array([1, 2, 3])
        result = buche_rastrigin.compute_s_i(z_i)

        expected = np.array([
            10 * (10 ** (0.5 * (0) / 2)),
            10 ** (0.5 * (1) / 2),
            10 * (10 ** (0.5 * (2) / 2))
        ])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


if __name__ == '__main__':
    unittest.main()
