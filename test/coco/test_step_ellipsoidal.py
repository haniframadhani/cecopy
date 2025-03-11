import unittest
import numpy as np
from ceco.bbob import Bbob
from ceco.coco.step_ellipsoidal import Step_ellipsoidal


class Test_ellipsoidal(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        step = Step_ellipsoidal(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(step.x_opt), dimension)
        self.assertTrue(np.all(step.x_opt >= -5)
                        and np.all(step.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = step.raw(step.x_opt)
        self.assertAlmostEqual(step.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(step.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        step = Step_ellipsoidal(dimension)

        # Evaluate at x_opt
        result = step.evaluate(step.x_opt)
        self.assertAlmostEqual(result, step.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        step = Step_ellipsoidal(dimension)
        bbob = Bbob(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = step.evaluate(input_vector)

        # Manually compute the expected result
        R = [[0.39134578, -0.92024371], [0.92024371,  0.39134578]]
        z_hat = np.matmul(bbob.create_diagonal_matrix(
            10), np.matmul(R, input_vector - step.x_opt))

        z_tilde = np.zeros_like(z_hat)
        for i in range(z_hat.shape[0]):
            if np.abs(z_hat[i]) > 0.5:
                z_tilde[i] = np.floor(0.5+z_hat[i])
            else:
                z_tilde[i] = np.floor(0.5+10*z_hat[i])/10

        Q = [[0.89942118, -0.43708299], [0.43708299,  0.89942118]]
        z = np.matmul(Q, z_tilde)

        term1 = np.abs(z_hat[0]) / 1e4

        exponents = 2 * (np.arange(dimension)-1)/(dimension - 1)
        weight = 10 ** exponents
        term2 = np.sum(weight * z ** 2)

        expected_result = 0.1 * max(term1, term2) + \
            bbob.f_pen(input_vector) + step.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        step = Step_ellipsoidal(dimension)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = step.evaluate(input_vector)

        # Manually compute the expected result
        R = [[0.39134578, -0.92024371], [0.92024371,  0.39134578]]
        z_hat = np.matmul(bbob.create_diagonal_matrix(
            10), np.matmul(R, input_vector - step.x_opt))

        z_tilde = np.zeros_like(z_hat)
        for i in range(z_hat.shape[0]):
            if np.abs(z_hat[i]) > 0.5:
                z_tilde[i] = np.floor(0.5+z_hat[i])
            else:
                z_tilde[i] = np.floor(0.5+10*z_hat[i])/10

        Q = [[0.89942118, -0.43708299], [0.43708299,  0.89942118]]
        z = np.matmul(Q, z_tilde)

        term1 = np.abs(z_hat[0]) / 1e4

        exponents = 2 * (np.arange(dimension)-1)/(dimension - 1)
        weight = 10 ** exponents
        term2 = np.sum(weight * z ** 2)

        expected_result = 0.1 * max(term1, term2) + \
            bbob.f_pen(input_vector) + step.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        step = Step_ellipsoidal(dimension)
        bbob = Bbob(dimension)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = step.evaluate(input_vector)

        # Manually compute the expected result
        R = [[-1.0]]
        z_hat = np.matmul(bbob.create_diagonal_matrix(
            10), np.matmul(R, input_vector - step.x_opt))

        z_tilde = np.zeros_like(z_hat)
        for i in range(z_hat.shape[0]):
            if np.abs(z_hat[i]) > 0.5:
                z_tilde[i] = np.floor(0.5+z_hat[i])
            else:
                z_tilde[i] = np.floor(0.5+10*z_hat[i])/10

        Q = [[1.0]]
        z = np.matmul(Q, z_tilde)

        term1 = np.abs(z_hat[0]) / 1e4

        exponents = 0
        weight = 10 ** exponents
        term2 = np.sum(weight * z ** 2)

        expected_result = 0.1 * max(term1, term2) + \
            bbob.f_pen(input_vector) + step.f_opt
        expected_result = 0.1 * max(term1, term2) + \
            bbob.f_pen(input_vector) + step.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        step = Step_ellipsoidal(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            step.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
