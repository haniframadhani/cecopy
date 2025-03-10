import unittest
from ceco.bbob import Bbob
from ceco.coco.rastrigin import Rastrigin
import numpy as np


class Test_rastrigin(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        rastrigin = Rastrigin(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(rastrigin.x_opt), dimension)
        self.assertTrue(np.all(rastrigin.x_opt >= -5)
                        and np.all(rastrigin.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = rastrigin.raw(rastrigin.x_opt)
        self.assertAlmostEqual(rastrigin.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(rastrigin.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

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
        input_vector = np.zeros(dimension)
        result = rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - rastrigin.x_opt
        z = bbob.T_osz(z)
        z = np.matmul(
            bbob.create_diagonal_matrix(10), bbob.T_asy_beta(0.2, z))
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        expected_result = 10 * (dimension - sum_cos)
        expected_result = expected_result + \
            bbob.euclidean_norm(z) + rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        rastrigin = Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - rastrigin.x_opt
        z = bbob.T_osz(z)
        z = np.matmul(
            bbob.create_diagonal_matrix(10), bbob.T_asy_beta(0.2, z))
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        expected_result = 10 * (dimension - sum_cos)
        expected_result = expected_result + \
            bbob.euclidean_norm(z) + rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        rastrigin = Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Evaluate at x_opt
        result = rastrigin.evaluate(rastrigin.x_opt)
        self.assertAlmostEqual(result, rastrigin.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - rastrigin.x_opt
        z = bbob.T_osz(z)
        z = np.matmul(
            bbob.create_diagonal_matrix(10), bbob.T_asy_beta(0.2, z))
        sum_cos = np.sum(np.cos(2 * np.pi * z))
        expected_result = 10 * (dimension - sum_cos)
        expected_result = expected_result + \
            bbob.euclidean_norm(z) + rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        rastrigin = Rastrigin(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            rastrigin.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
