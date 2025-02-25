import unittest
import random
from ceco.coco.sphere import Sphere


class Test_sphere(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_initialization(self):
        dimension = 5
        sphere = Sphere(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(sphere.x_opt), dimension)
        for x in sphere.x_opt:
            self.assertTrue(-5 <= x <= 5)

        # Check if f_opt is computed correctly
        expected_f_opt = sphere.raw(sphere.x_opt)
        self.assertEqual(sphere.f_opt, expected_f_opt)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = [1.3942679845788373, -4.74989244777333, -
                          2.2497068163088074, -2.7678926185117723, 2.3647121416401244]
        self.assertEqual(sphere.x_opt, expected_x_opt)

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
        input_vector = [0.0] * dimension
        result = sphere.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, sphere.x_opt)]
        expected_result = sum(zi ** 2 for zi in z) + sphere.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        sphere = Sphere(dimension)

        # Input vector with negative values
        input_vector = [-3, -2]
        result = sphere.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, sphere.x_opt)]
        expected_result = sum(zi ** 2 for zi in z) + sphere.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_custom_f_opt(self):
        dimension = 3
        custom_f_opt = 10.0
        sphere = Sphere(dimension, f_opt=custom_f_opt)

        # Evaluate at x_opt
        result = sphere.evaluate(sphere.x_opt)
        self.assertAlmostEqual(result, custom_f_opt, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        sphere = Sphere(dimension)

        # Evaluate at x_opt
        result = sphere.evaluate(sphere.x_opt)
        self.assertAlmostEqual(result, sphere.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = [2.0]
        result = sphere.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, sphere.x_opt)]
        expected_result = sum(zi ** 2 for zi in z) + sphere.f_opt

        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        sphere = Sphere(dimension)

        # Empty input vector
        input_vector = []
        with self.assertRaises(ValueError):
            sphere.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
