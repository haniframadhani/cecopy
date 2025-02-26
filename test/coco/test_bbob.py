import unittest
import math
from ceco.bbob import Bbob  # Assuming the Bbob class is in a file named bbob.py


class TestBbob(unittest.TestCase):
    """
    Unit tests for the Bbob class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.dimension = 3
        self.bbob = Bbob(dimension=self.dimension)

    # Test cases for euclidean_norm
    def test_euclidean_norm(self):
        """
        Test the euclidean_norm method with a valid input vector.
        """
        input_vector = [3, 4, 0]
        expected_norm = 5  # sqrt(3^2 + 4^2 + 0^2) = 5
        self.assertAlmostEqual(
            self.bbob.euclidean_norm(input_vector), expected_norm)

    def test_euclidean_norm_empty_vector(self):
        """
        Test the euclidean_norm method with an empty input vector.
        """
        input_vector = []
        with self.assertRaises(ValueError):
            self.bbob.euclidean_norm(input_vector)

    def test_euclidean_norm_single_element(self):
        """
        Test the euclidean_norm method with a single-element vector.
        """
        input_vector = [5]
        expected_norm = 5  # sqrt(5^2) = 5
        self.assertAlmostEqual(
            self.bbob.euclidean_norm(input_vector), expected_norm)

    # Test cases for create_diagonal_matrix
    def test_create_diagonal_matrix(self):
        """
        Test the create_diagonal_matrix method with valid parameters.
        """
        alpha = 2
        expected_matrix = [
            [2 ** 0, 0, 0],
            [0, 2 ** (0.5 * (1 / (self.dimension - 1))), 0],
            [0, 0, 2 ** (0.5 * (2 / (self.dimension - 1)))]
        ]
        result_matrix = self.bbob.create_diagonal_matrix(alpha)
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.assertAlmostEqual(
                    result_matrix[i][j], expected_matrix[i][j])

    def test_create_diagonal_matrix_dimension_one(self):
        """
        Test the create_diagonal_matrix method when dimension = 1.
        """
        bbob = Bbob(dimension=1)
        alpha = 2
        expected_matrix = [[2 ** 0]]
        result_matrix = bbob.create_diagonal_matrix(alpha)
        self.assertEqual(result_matrix, expected_matrix)

    def test_create_diagonal_matrix_zero_alpha(self):
        """
        Test the create_diagonal_matrix method with alpha = 0.
        """
        alpha = 0
        expected_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        result_matrix = self.bbob.create_diagonal_matrix(alpha)
        self.assertEqual(result_matrix, expected_matrix)

    # Test cases for T_asy_beta
    def test_T_asy_beta(self):
        """
        Test the T_asy_beta method with valid input.
        """
        beta = 1
        input_vector = [1, 2, 3]
        expected_output = [
            1 ** (1 + 1 * 0 / 2 * math.sqrt(1)),
            2 ** (1 + 1 * 1 / 2 * math.sqrt(2)),
            3 ** (1 + 1 * 2 / 2 * math.sqrt(3))
        ]
        result = self.bbob.T_asy_beta(beta, input_vector)
        for i in range(self.dimension):
            self.assertAlmostEqual(result[i], expected_output[i])

    def test_T_asy_beta_dimension_one(self):
        """
        Test the T_asy_beta method when dimension = 1.
        """
        bbob = Bbob(dimension=1)
        beta = 1
        input_vector = [2]
        expected_output = [2]  # No transformation for D = 1
        result = bbob.T_asy_beta(beta, input_vector)
        self.assertEqual(result, expected_output)

    def test_T_asy_beta_negative_values(self):
        """
        Test the T_asy_beta method with negative values in the input vector.
        """
        beta = 1
        input_vector = [-1, -2, -3]
        expected_output = [-1, -2, -3]  # Negative values remain unchanged
        result = self.bbob.T_asy_beta(beta, input_vector)
        self.assertEqual(result, expected_output)

    # Test cases for T_osz
    def test_T_osz(self):
        """
        Test the T_osz method with valid input.
        """
        input_vector = [1, -1, 0]
        result = self.bbob.T_osz(input_vector)

        # Expected results
        x1 = 1
        x1_hat = math.log(abs(x1))
        c1, c2 = 10, 7.9
        sine_term1 = 0.049 * (math.sin(c1 * x1_hat) + math.sin(c2 * x1_hat))
        expected1 = math.exp(x1_hat + sine_term1)

        x2 = -1
        x2_hat = math.log(abs(x2))
        c1, c2 = 5.5, 3.1
        sine_term2 = 0.049 * (math.sin(c1 * x2_hat) + math.sin(c2 * x2_hat))
        expected2 = -math.exp(x2_hat + sine_term2)

        expected3 = 0  # For x = 0

        self.assertAlmostEqual(result[0], expected1)
        self.assertAlmostEqual(result[1], expected2)
        self.assertAlmostEqual(result[2], expected3)

    def test_T_osz_zero_vector(self):
        """
        Test the T_osz method with a vector of zeros.
        """
        input_vector = [0, 0, 0]
        expected_output = [0, 0, 0]
        result = self.bbob.T_osz(input_vector)
        self.assertEqual(result, expected_output)

    def test_T_osz_negative_values(self):
        """
        Test the T_osz method with negative values in the input vector.
        """
        input_vector = [-2, -3, -4]
        result = self.bbob.T_osz(input_vector)
        for x in result:
            self.assertLessEqual(x, 0)  # All transformed values should be <= 0

    # Test cases for elementwise_multiply
    def test_elementwise_multiply(self):
        """
        Test the elementwise_multiply method with valid input.
        """
        x = [1, 2, 3]
        y = [4, 5, 6]
        expected_result = [1 * 4, 2 * 5, 3 * 6]
        result = self.bbob.elementwise_multiply(x, y)
        self.assertEqual(result, expected_result)

    def test_elementwise_multiply_invalid_input(self):
        """
        Test the elementwise_multiply method with invalid input (vectors of different lengths).
        """
        x = [1, 2, 3]
        y = [4, 5]  # Different length
        with self.assertRaises(ValueError):
            self.bbob.elementwise_multiply(x, y)

    def test_elementwise_multiply_empty_vectors(self):
        """
        Test the elementwise_multiply method with empty vectors.
        """
        x = []
        y = []
        expected_result = []
        result = self.bbob.elementwise_multiply(x, y)
        self.assertEqual(result, expected_result)

    def test_matrix_multiply_2d_matrices(self):
        """
        Test matrix multiplication with two 2D matrices.
        """
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        expected_result = [[19, 22], [43, 50]]  # Computed manually
        result = self.bbob.matrix_multiply(A, B)
        self.assertEqual(result, expected_result)

    def test_matrix_multiply_2d_matrix_1d_vector(self):
        """
        Test matrix multiplication with a 2D matrix and a 1D vector.
        """
        A = [[1, 2], [3, 4]]
        B = [5, 6]
        expected_result = [17, 39]  # Computed as [1*5 + 2*6, 3*5 + 4*6]
        result = self.bbob.matrix_multiply(A, B)
        self.assertEqual(result, expected_result)

    def test_matrix_multiply_1d_vectors(self):
        """
        Test dot product of two 1D vectors.
        """
        A = [1, 2, 3]
        B = [4, 5, 6]
        expected_result = 32  # Computed as 1*4 + 2*5 + 3*6
        result = self.bbob.matrix_multiply(A, B)
        self.assertEqual(result, expected_result)

    def test_matrix_multiply_incompatible_dimensions(self):
        """
        Test matrix multiplication with incompatible dimensions (should raise ValueError).
        """
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10]]  # Incorrect shape
        with self.assertRaises(ValueError):
            self.bbob.matrix_multiply(A, B)


if __name__ == "__main__":
    unittest.main()
