import unittest
import random
from ceco.bbob import Bbob
from ceco.coco.buche_rastrigin import Buche_Rastrigin
import math


class Test_buche_rastrigin(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_initialization(self):
        dimension = 5
        buche_rastrigin = Buche_Rastrigin(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(buche_rastrigin.x_opt), dimension)
        for x in buche_rastrigin.x_opt:
            self.assertTrue(-5 <= x <= 5)

        # Check if f_opt is computed correctly
        expected_f_opt = buche_rastrigin.raw(buche_rastrigin.x_opt)
        self.assertEqual(buche_rastrigin.f_opt, expected_f_opt)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = [1.3942679845788373, -4.74989244777333, -
                          2.2497068163088074, -2.7678926185117723, 2.3647121416401244]
        self.assertEqual(buche_rastrigin.x_opt, expected_x_opt)

    def test_evaluate_at_optimal_point(self):
        dimension = 3
        buche_rastrigin = Buche_Rastrigin(dimension)

        # Evaluate at x_opt
        result = buche_rastrigin.evaluate(buche_rastrigin.x_opt)
        self.assertAlmostEqual(result, buche_rastrigin.f_opt, places=6)

    def test_evaluate_at_zero_vector(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 2
        buche_rastrigin = Buche_Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Zero vector
        input_vector = [0.0] * dimension
        result = buche_rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, buche_rastrigin.x_opt)]
        z = bbob.T_osz(z)
        s = bbob.compute_s_i(z)
        z = [x * y for x, y in zip(z, s)]
        sum_cos = sum(math.cos(2 * math.pi * x_i) for x_i in z)
        sum_square = sum(x_i ** 2 for x_i in z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            bbob.f_pen(input_vector) + buche_rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        buche_rastrigin = Buche_Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = [-3, -2]
        result = buche_rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, buche_rastrigin.x_opt)]
        z = bbob.T_osz(z)
        s = bbob.compute_s_i(z)
        z = [x * y for x, y in zip(z, s)]
        sum_cos = sum(math.cos(2 * math.pi * x_i) for x_i in z)
        sum_square = sum(x_i ** 2 for x_i in z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            bbob.f_pen(input_vector) + buche_rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_custom_f_opt(self):
        dimension = 3
        custom_f_opt = 10.0
        buche_rastrigin = Buche_Rastrigin(dimension, f_opt=custom_f_opt)

        # Evaluate at x_opt
        result = buche_rastrigin.evaluate(buche_rastrigin.x_opt)
        self.assertAlmostEqual(result, custom_f_opt, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        buche_rastrigin = Buche_Rastrigin(dimension)
        bbob = Bbob(dimension)

        # Evaluate at x_opt
        result = buche_rastrigin.evaluate(buche_rastrigin.x_opt)
        self.assertAlmostEqual(result, buche_rastrigin.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = [2.0]
        result = buche_rastrigin.evaluate(input_vector)

        # Manually compute the expected result
        z = [x - y for x, y in zip(input_vector, buche_rastrigin.x_opt)]
        z = bbob.T_osz(z)
        s = bbob.compute_s_i(z)
        z = [x * y for x, y in zip(z, s)]
        sum_cos = sum(math.cos(2 * math.pi * x_i) for x_i in z)
        sum_square = sum(x_i ** 2 for x_i in z)
        expected_result = 10 * (dimension - sum_cos) + sum_square + 100
        expected_result = expected_result * \
            bbob.f_pen(input_vector) + buche_rastrigin.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        buche_rastrigin = Buche_Rastrigin(dimension)

        # Empty input vector
        input_vector = []
        with self.assertRaises(ValueError):
            buche_rastrigin.evaluate(input_vector)


if __name__ == '__main__':
    unittest.main()
