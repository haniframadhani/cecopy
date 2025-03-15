import unittest
from ceco.bbob import Bbob
from ceco.coco.schwefel import Schwefel
import numpy as np


class Test_schwefel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        schwefel = Schwefel(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(schwefel.x_opt), dimension)
        self.assertTrue(np.all(schwefel.x_opt >= -5)
                        and np.all(schwefel.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = schwefel.raw(schwefel.x_opt)
        self.assertAlmostEqual(schwefel.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(schwefel.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    # def test_evaluate_at_optimal_point(self):
    #     dimension = 3
    #     schwefel = Schwefel(dimension)

    #     # Evaluate at x_opt
    #     result = schwefel.evaluate(schwefel.x_opt)
    #     self.assertAlmostEqual(result, schwefel.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        schwefel = Schwefel(dimension)
        bbob = Bbob(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = schwefel.evaluate(input_vector)

        # Manually compute the expected result
        # 1. Apply first transformation: x_hat = 2 × 1± ⊗ x
        x_hat = 2 * schwefel.sign_vector * input_vector

        # 2. Calculate z_hat values
        z_hat = np.zeros(dimension)
        z_hat[0] = x_hat[0]  # z_hat_1 = x_hat_1

        for i in range(dimension - 1):
            # Calculate the term [x_i^opt → 2|x_i^opt|]
            x_opt_term = 2 * np.abs(schwefel.true_x_opt[i])

            # Calculate z_hat_{i+1} using the recursive formula
            z_hat[i+1] = x_hat[i+1] + 0.25 * (x_hat[i] - x_opt_term)

        # 3. Prepare the final transformation for z
        # Calculate the vector [x^opt → 2|x^opt|]
        x_opt_transformed = 2 * np.abs(schwefel.true_x_opt)

        # Calculate the difference: z_hat - [x^opt → 2|x^opt|]
        diff_vector = z_hat - x_opt_transformed

        # Apply the diagonal matrix A^10
        lambda_matrix = bbob.create_diagonal_matrix(10)

        # Multiply the diagonal matrix by the difference vector
        matrix_result = np.matmul(lambda_matrix, diff_vector)

        # Add 2|x^opt| to the result and scale by 100
        z = 100 * (matrix_result + 2 * np.abs(schwefel.true_x_opt))

        # 4. Calculate the penalty term: 100*f_pen(z/100)
        z_scaled = z/100
        penalty = 100 * bbob.f_pen(z_scaled)

        # 5. Calculate the raw Schwefel function value
        raw_value = schwefel.raw(input_vector)

        # 6. Combine all terms to get the final result
        expected_result = raw_value + 4.189828872724339 + penalty + schwefel.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        schwefel = Schwefel(dimension)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = schwefel.evaluate(input_vector)

        # Manually compute the expected result
        # 1. Apply first transformation: x_hat = 2 × 1± ⊗ x
        x_hat = 2 * schwefel.sign_vector * input_vector

        # 2. Calculate z_hat values
        z_hat = np.zeros(dimension)
        z_hat[0] = x_hat[0]  # z_hat_1 = x_hat_1

        for i in range(dimension - 1):
            # Calculate the term [x_i^opt → 2|x_i^opt|]
            x_opt_term = 2 * np.abs(schwefel.true_x_opt[i])

            # Calculate z_hat_{i+1} using the recursive formula
            z_hat[i+1] = x_hat[i+1] + 0.25 * (x_hat[i] - x_opt_term)

        # 3. Prepare the final transformation for z
        # Calculate the vector [x^opt → 2|x^opt|]
        x_opt_transformed = 2 * np.abs(schwefel.true_x_opt)

        # Calculate the difference: z_hat - [x^opt → 2|x^opt|]
        diff_vector = z_hat - x_opt_transformed

        # Apply the diagonal matrix A^10
        lambda_matrix = bbob.create_diagonal_matrix(10)

        # Multiply the diagonal matrix by the difference vector
        matrix_result = np.matmul(lambda_matrix, diff_vector)

        # Add 2|x^opt| to the result and scale by 100
        z = 100 * (matrix_result + 2 * np.abs(schwefel.true_x_opt))

        # 4. Calculate the penalty term: 100*f_pen(z/100)
        z_scaled = z/100
        penalty = 100 * bbob.f_pen(z_scaled)

        # 5. Calculate the raw Schwefel function value
        raw_value = schwefel.raw(input_vector)

        # 6. Combine all terms to get the final result
        expected_result = raw_value + 4.189828872724339 + penalty + schwefel.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        schwefel = Schwefel(dimension)
        bbob = Bbob(dimension)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = schwefel.evaluate(input_vector)

        # Manually compute the expected result
        # 1. Apply first transformation: x_hat = 2 × 1± ⊗ x
        x_hat = 2 * schwefel.sign_vector * input_vector

        # 2. Calculate z_hat values
        z_hat = np.zeros(dimension)
        z_hat[0] = x_hat[0]  # z_hat_1 = x_hat_1

        for i in range(dimension - 1):
            # Calculate the term [x_i^opt → 2|x_i^opt|]
            x_opt_term = 2 * np.abs(schwefel.true_x_opt[i])

            # Calculate z_hat_{i+1} using the recursive formula
            z_hat[i+1] = x_hat[i+1] + 0.25 * (x_hat[i] - x_opt_term)

        # 3. Prepare the final transformation for z
        # Calculate the vector [x^opt → 2|x^opt|]
        x_opt_transformed = 2 * np.abs(schwefel.true_x_opt)

        # Calculate the difference: z_hat - [x^opt → 2|x^opt|]
        diff_vector = z_hat - x_opt_transformed

        # Apply the diagonal matrix A^10
        lambda_matrix = bbob.create_diagonal_matrix(10)

        # Multiply the diagonal matrix by the difference vector
        matrix_result = np.matmul(lambda_matrix, diff_vector)

        # Add 2|x^opt| to the result and scale by 100
        z = 100 * (matrix_result + 2 * np.abs(schwefel.true_x_opt))

        # 4. Calculate the penalty term: 100*f_pen(z/100)
        z_scaled = z/100
        penalty = 100 * bbob.f_pen(z_scaled)

        # 5. Calculate the raw Schwefel function value
        raw_value = schwefel.raw(input_vector)

        # 6. Combine all terms to get the final result
        expected_result = raw_value + 4.189828872724339 + penalty + schwefel.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        schwefel = Schwefel(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            schwefel.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
