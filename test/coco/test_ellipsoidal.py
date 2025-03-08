import unittest
import numpy as np
from ceco.bbob import Bbob
from ceco.coco.ellipsoidal import Ellipsoidal


class Test_ellipsoidal(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        ellipsoidal = Ellipsoidal(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(ellipsoidal.x_opt), dimension)
        self.assertTrue(np.all(ellipsoidal.x_opt >= -5)
                        and np.all(ellipsoidal.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = ellipsoidal.raw(ellipsoidal.x_opt)
        self.assertAlmostEqual(ellipsoidal.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(ellipsoidal.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        ellipsoidal = Ellipsoidal(dimension)

        # Evaluate at x_opt
        result = ellipsoidal.evaluate(ellipsoidal.x_opt)
        self.assertAlmostEqual(result, ellipsoidal.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        ellipsoidal = Ellipsoidal(dimension)
        bbob = Bbob(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = ellipsoidal.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - ellipsoidal.x_opt
        z = bbob.T_osz(z)
        i = np.arange(1, dimension + 1)
        exponent = 6 * (i - 1) / (dimension - 1)
        expected_result = np.sum((10 ** exponent) * (z ** 2))
        expected_result = expected_result + ellipsoidal.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        ellipsoidal = Ellipsoidal(dimension)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = ellipsoidal.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - ellipsoidal.x_opt
        z = bbob.T_osz(z)
        i = np.arange(1, dimension + 1)
        exponent = 6 * (i - 1) / (dimension - 1)
        expected_result = np.sum((10 ** exponent) * (z ** 2))
        expected_result = expected_result + ellipsoidal.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_custom_f_opt(self):
        dimension = 3
        custom_f_opt = 10.0
        ellipsoidal = Ellipsoidal(dimension, f_opt=custom_f_opt)

        # Evaluate at x_opt
        result = ellipsoidal.evaluate(ellipsoidal.x_opt)
        self.assertAlmostEqual(result, custom_f_opt, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        ellipsoidal = Ellipsoidal(dimension)
        bbob = Bbob(dimension)

        # Evaluate at x_opt
        result = ellipsoidal.evaluate(ellipsoidal.x_opt)
        self.assertAlmostEqual(result, ellipsoidal.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = ellipsoidal.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - ellipsoidal.x_opt
        z = bbob.T_osz(z)
        exponent = 0
        expected_result = np.sum((10 ** exponent) * (z ** 2))
        expected_result = expected_result + ellipsoidal.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        ellipsoidal = Ellipsoidal(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            ellipsoidal.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
