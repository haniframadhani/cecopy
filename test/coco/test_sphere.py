import unittest
import numpy as np
from ceco.coco.sphere import Sphere


class Test_sphere(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        sphere = Sphere(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(sphere.x_opt), dimension)
        self.assertTrue(np.all(sphere.x_opt >= -5)
                        and np.all(sphere.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = sphere.raw(sphere.x_opt)
        self.assertAlmostEqual(sphere.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(sphere.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        sphere = Sphere(dimension)

        # Evaluate at x_opt
        result = sphere.evaluate(sphere.x_opt)
        self.assertAlmostEqual(result, sphere.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        sphere = Sphere(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        result = sphere.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - sphere.x_opt
        expected_result = np.sum(z ** 2) + sphere.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        sphere = Sphere(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = sphere.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - sphere.x_opt
        expected_result = np.sum(z ** 2) + sphere.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        sphere = Sphere(dimension)

        # Evaluate at x_opt
        result = sphere.evaluate(sphere.x_opt)
        self.assertAlmostEqual(result, sphere.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = sphere.evaluate(input_vector)

        # Manually compute the expected result
        z = input_vector - sphere.x_opt
        expected_result = np.sum(z ** 2) + sphere.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        sphere = Sphere(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            sphere.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
