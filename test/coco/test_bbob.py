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

    def test_gram_schmidt_orthogonalization(self):
        """
            Test if the output vectors are orthogonal.
            """
        # Input matrix with linearly independent vectors
        A = np.array([[1, 1, 1],
                      [0, 1, 1],
                      [0, 0, 1]], dtype=float)

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that all pairs of columns are orthogonal
        for i in range(Q.shape[1]):
            for j in range(i + 1, Q.shape[1]):
                dot_product = np.dot(Q[:, i], Q[:, j])
                self.assertTrue(np.isclose(dot_product, 0, atol=1e-10),
                                f"Vectors {i} and {j} are not orthogonal.")

    def test_gram_schmidt_normalization(self):
        """
        Test if the output vectors are normalized (unit length).
        """
        # Input matrix with linearly independent vectors
        A = np.array([[1, 2, 3],
                      [0, 1, 2],
                      [0, 0, 1]], dtype=float)

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that each column vector has unit length
        for i in range(Q.shape[1]):
            norm = np.linalg.norm(Q[:, i])
            self.assertTrue(np.isclose(norm, 1, atol=1e-10),
                            f"Vector {i} is not normalized.")

    def test_gram_schmidt_linearly_dependent_vectors(self):
        """
        Test behavior with linearly dependent vectors.
        """
        # Input matrix with linearly dependent vectors
        A = np.array([[1, 2, 3],
                      [2, 4, 6],
                      [3, 6, 9]], dtype=float)

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that the last vector is zero (due to linear dependence)
        self.assertTrue(np.allclose(Q[:, 2], np.zeros(Q.shape[0])),
                        "Linearly dependent vectors were not handled correctly.")

    def test_gram_schmidt_identity_matrix(self):
        """
        Test with an identity matrix (already orthogonal and normalized).
        """
        # Input matrix is the identity matrix
        A = np.eye(3)

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that the output is the same as the input
        self.assertTrue(np.allclose(Q, A),
                        "Identity matrix was not preserved.")

    def test_gram_schmidt_zero_matrix(self):
        """
        Test with a zero matrix (all vectors are zero).
        """
        # Input matrix is a zero matrix
        A = np.zeros((3, 3))

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that the output is also a zero matrix
        self.assertTrue(np.allclose(Q, A),
                        "Zero matrix was not preserved.")

    def test_gram_schmidt_non_square_matrix(self):
        """
        Test with a non-square matrix.
        """
        # Input matrix with more columns than rows
        A = np.array([[1, 2, 3],
                      [0, 1, 2]], dtype=float)

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that the output vectors are orthogonal and normalized
        for i in range(Q.shape[1]):
            for j in range(i + 1, Q.shape[1]):
                dot_product = np.dot(Q[:, i], Q[:, j])
                self.assertTrue(np.isclose(dot_product, 0, atol=1e-10),
                                f"Vectors {i} and {j} are not orthogonal.")

            # Check if the vector is effectively zero (due to linear dependence)
            norm = np.linalg.norm(Q[:, i])
            if np.isclose(norm, 0, atol=1e-10):
                # If the vector is zero, skip normalization check
                self.assertTrue(np.allclose(Q[:, i], np.zeros(Q.shape[0])),
                                f"Vector {i} is not zero.")
            else:
                # Otherwise, check that the vector is normalized
                self.assertTrue(np.isclose(norm, 1, atol=1e-10),
                                f"Vector {i} is not normalized.")

    def test_gram_schmidt_matrix_values_non_orthogonal(self):
        """
        Test the Gram-Schmidt process with a non-orthogonal input matrix.
        """
        # Input matrix (3x3)
        A = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [0, 1, 1]], dtype=float)

        # Expected orthogonalized matrix (precomputed using a trusted implementation)
        expected_Q = np.array([[0.70710678, 0.40824829, -0.57735027],
                               [0.70710678, -0.40824829, 0.57735027],
                               [0.0, 0.81649658, 0.57735027]], dtype=float)

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that the output matches the expected result
        self.assertTrue(np.allclose(Q, expected_Q, atol=1e-8),
                        "Output matrix does not match the expected result.")

    def test_gram_schmidt_bigger_dimension(self):
        """
        Test the Gram-Schmidt process with a non-orthogonal input matrix.
        """
        # Input matrix (3x3)
        A = np.array([[1, 1, 1, 0, 0],
                      [1, 0, 1, 0, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=float).T

        expected_Q = np.array([[0.57735027, 0.40824829, -0.70710678, 0, 0],
                               [0.57735027, -0.81649658, 0.0, 0, 0],
                               [0.57735027, 0.40824829, 0.70710678, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]], dtype=float)

        # Apply Gram-Schmidt
        Q = self.bbob.gram_schmidt(A)

        # Verify that the output matches the expected result
        self.assertTrue(np.allclose(Q, expected_Q, atol=1e-8),
                        "Output matrix does not match the expected result.")


if __name__ == "__main__":
    unittest.main()
