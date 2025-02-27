import unittest
import random
from ceco.bbob import Bbob
from ceco.coco.rosenbrock import Rosenbrock
import math


class Test_rosenbrock(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_initialization(self):
        dimension = 5
        rosenbrock = Rosenbrock(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(rosenbrock.x_opt), dimension)
        for x in rosenbrock.x_opt:
            self.assertTrue(-5 <= x <= 5)

        # Check if f_opt is computed correctly
        expected_f_opt = rosenbrock.raw(rosenbrock.x_opt)
        self.assertEqual(rosenbrock.f_opt, expected_f_opt)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = [1.3942679845788373, -4.74989244777333, -
                          2.2497068163088074, -2.7678926185117723, 2.3647121416401244]
        self.assertEqual(rosenbrock.x_opt, expected_x_opt)

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        rosenbrock = Rosenbrock(dimension)

        # Evaluate at x_opt
        result = rosenbrock.evaluate(rosenbrock.x_opt)
        self.assertAlmostEqual(result, rosenbrock.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        rosenbrock = Rosenbrock(dimension)
        bbob = Bbob(dimension)

        # Zero vector
        input_vector = [0.0] * dimension
        result = rosenbrock.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = [scaling_factor * (input_vector[i]-rosenbrock.x_opt[i]) +
             1 for i in range(rosenbrock.dimension)]
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        rosenbrock = Rosenbrock(dimension)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = [-3, -2]
        result = rosenbrock.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = [scaling_factor * (input_vector[i]-rosenbrock.x_opt[i]) +
             1 for i in range(rosenbrock.dimension)]
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_custom_f_opt(self):
        dimension = 3
        custom_f_opt = 10.0
        rosenbrock = Rosenbrock(dimension, f_opt=custom_f_opt)

        # Evaluate at x_opt
        result = rosenbrock.evaluate(rosenbrock.x_opt)
        self.assertAlmostEqual(result, custom_f_opt, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        rosenbrock = Rosenbrock(dimension)
        bbob = Bbob(dimension)

        # Evaluate at x_opt
        result = rosenbrock.evaluate(rosenbrock.x_opt)
        self.assertAlmostEqual(result, rosenbrock.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = [2.0]
        result = rosenbrock.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = [scaling_factor * (input_vector[i]-rosenbrock.x_opt[i]) +
             1 for i in range(rosenbrock.dimension)]
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        rosenbrock = Rosenbrock(dimension)

        # Empty input vector
        input_vector = []
        with self.assertRaises(ValueError):
            rosenbrock.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
