import unittest
import numpy as np
from ceco.bbob import Bbob


class TestBbob(unittest.TestCase):
    """
    Unit tests for the Bbob class.
    """

    def setUp(self):
        self.dimension = 3
        self.bbob = Bbob(dimension=self.dimension)

    # Test Euclidean Norm
    def test_euclidean_norm(self):
        input_vector = np.array([3, 4, 0])
        expected = 5.0  # sqrt(3^2 + 4^2)
        result = self.bbob.euclidean_norm(input_vector)
        self.assertAlmostEqual(result, expected)

    def test_euclidean_norm_empty_vector(self):
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            self.bbob.euclidean_norm(input_vector)

    def test_euclidean_norm_single_element(self):
        input_vector = np.array([5])
        expected = 5.0
        result = self.bbob.euclidean_norm(input_vector)
        self.assertAlmostEqual(result, expected)

    # Test Diagonal Matrix
    def test_create_diagonal_matrix(self):
        alpha = 2
        expected = np.diag([2**0, 2**(0.5 * 1 / 2), 2**(0.5 * 2 / 2)])
        result = self.bbob.create_diagonal_matrix(alpha)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_create_diagonal_matrix_dimension_one(self):
        bbob = Bbob(dimension=1)
        alpha = 2
        expected = np.array([[1.0]])
        result = bbob.create_diagonal_matrix(alpha)
        np.testing.assert_array_almost_equal(result, expected)

    def test_create_diagonal_matrix_zero_alpha(self):
        alpha = 0
        expected = np.zeros((self.dimension, self.dimension))
        result = self.bbob.create_diagonal_matrix(alpha)
        np.testing.assert_array_almost_equal(result, expected)

    # Test Random Matrix
    def test_generate_random_matrix(self):
        matrix = self.bbob.generate_random_matrix(self.dimension)
        self.assertEqual(matrix.shape, (self.dimension, self.dimension))

    # Test T_asy_beta
    def test_T_asy_beta(self):
        beta = 1.0
        input_vector = np.array([1, 2, 3])
        expected = np.array([
            1 ** (1 + beta * 0 / 2 * np.sqrt(1)),
            2 ** (1 + beta * 1 / 2 * np.sqrt(2)),
            3 ** (1 + beta * 2 / 2 * np.sqrt(3))
        ])
        result = self.bbob.T_asy_beta(beta, input_vector)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_T_asy_beta_negative(self):
        beta = 1.0
        input_vector = np.array([-1, -2, -3])
        expected = input_vector
        result = self.bbob.T_asy_beta(beta, input_vector)
        np.testing.assert_array_equal(result, expected)

    # Test T_osz
    def test_T_osz(self):
        input_vector = np.array([1, -1, 0])
        result = self.bbob.T_osz(input_vector)

        expected = np.array([
            np.exp(np.log(1) + 0.049 *
                   (np.sin(10 * np.log(1)) + np.sin(7.9 * np.log(1)))),
            -np.exp(np.log(1) + 0.049 *
                    (np.sin(5.5 * np.log(1)) + np.sin(3.1 * np.log(1)))),
            0
        ])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_T_osz_zero_vector(self):
        input_vector = np.array([0, 0, 0])
        expected = np.array([0, 0, 0])
        result = self.bbob.T_osz(input_vector)
        np.testing.assert_array_equal(result, expected)

    # Test Element-wise Multiplication
    def test_elementwise_multiply(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        expected = np.array([4, 10, 18])
        result = self.bbob.elementwise_multiply(x, y)
        np.testing.assert_array_equal(result, expected)

    def test_elementwise_multiply_invalid(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        with self.assertRaises(ValueError):
            self.bbob.elementwise_multiply(x, y)

    def test_f_pen_no_penalty(self):
        x = np.array([1, 2, 3])
        result = self.bbob.f_pen(x)
        self.assertAlmostEqual(result, 0.0)

    def test_f_pen_partial_penalty(self):
        x = np.array([1, 6, 3])
        result = self.bbob.f_pen(x)
        self.assertAlmostEqual(result, 1.0)

    def test_f_pen_all_penalty(self):
        x = np.array([6, 7, 8])
        result = self.bbob.f_pen(x)
        self.assertAlmostEqual(result, 14.0)


if __name__ == "__main__":
    unittest.main()
