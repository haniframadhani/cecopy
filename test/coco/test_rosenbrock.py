import unittest
from ceco.coco.rosenbrock import Rosenbrock
from ceco.bbob import Bbob
import numpy as np


class Test_rosenbrock(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_initialization(self):
        dimension = 5
        rosenbrock = Rosenbrock(dimension)

        # Check if x_opt is generated correctly
        self.assertEqual(len(rosenbrock.x_opt), dimension)
        self.assertTrue(np.all(rosenbrock.x_opt >= -5)
                        and np.all(rosenbrock.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = rosenbrock.raw(rosenbrock.x_opt)
        self.assertAlmostEqual(rosenbrock.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(rosenbrock.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

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

        # Zero vector
        input_vector = np.zeros(dimension)
        result = rosenbrock.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = scaling_factor * (input_vector - rosenbrock.x_opt) + 1
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values(self):
        dimension = 2
        rosenbrock = Rosenbrock(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2])
        result = rosenbrock.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = scaling_factor * (input_vector - rosenbrock.x_opt) + 1
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1(self):
        dimension = 1
        rosenbrock = Rosenbrock(dimension)

        # Evaluate at x_opt
        result = rosenbrock.evaluate(rosenbrock.x_opt)
        self.assertAlmostEqual(result, rosenbrock.f_opt, places=6)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        result = rosenbrock.evaluate(input_vector)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = scaling_factor * (input_vector - rosenbrock.x_opt) + 1
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector(self):
        dimension = 3
        rosenbrock = Rosenbrock(dimension)

        # Empty input vector
        input_vector = np.array([])
        with self.assertRaises(ValueError):
            rosenbrock.evaluate(input_vector)

    def test_initialization_rotated(self):
        dimension = 5
        rosenbrock = Rosenbrock(dimension, rotation=True)

        # Check if x_opt is generated correctly
        self.assertEqual(len(rosenbrock.x_opt), dimension)
        self.assertTrue(np.all(rosenbrock.x_opt >= -5)
                        and np.all(rosenbrock.x_opt <= 5))

        # Check if f_opt is computed correctly
        expected_f_opt = rosenbrock.raw(rosenbrock.x_opt)
        self.assertAlmostEqual(rosenbrock.f_opt, expected_f_opt, places=6)

        # Check the specific values of x_opt for reproducibility
        expected_x_opt = np.array(
            [-1.25459881, 4.50714306, 2.31993942, 0.98658484, -3.4398136])
        self.assertTrue(np.allclose(rosenbrock.x_opt,
                        expected_x_opt, rtol=1e-6, atol=1e-6))

    # def test_evaluate_at_optimal_point_rotated(self):
    #     dimension = 3
    #     rosenbrock = Rosenbrock(dimension, rotation=True)

    #     # Evaluate at x_opt
    #     result = rosenbrock.evaluate(rosenbrock.x_opt, np.array([
    #         [0.20076996547107095, 0.04246242617649648, 0.9787177138112657],
    #         [-0.4005763178669152, 0.9152789497892921, 0.042462426176496426],
    #         [-0.8939966636005579, -0.4005763178669152, 0.20076996547107095]
    #     ]).T)
    #     self.assertAlmostEqual(result, rosenbrock.f_opt, places=6)

    def test_evaluate_at_zero_vector_rotated(self):
        """
        Test the evaluate method at the zero vector.
        """
        dimension = 3
        rosenbrock = Rosenbrock(dimension, rotation=True)
        bbob = Bbob(dimension)

        # Zero vector
        input_vector = np.zeros(dimension)
        rotation = np.array([
            [0.20076996547107095, 0.04246242617649648, 0.9787177138112657],
            [-0.4005763178669152, 0.9152789497892921, 0.042462426176496426],
            [-0.8939966636005579, -0.4005763178669152, 0.20076996547107095]
        ])
        result = rosenbrock.evaluate(input_vector, rotation)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = scaling_factor * \
            np.matmul(bbob.gram_schmidt(rotation.T), input_vector) + 1 / 2
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_negative_values_rotated(self):
        dimension = 3
        rosenbrock = Rosenbrock(dimension, rotation=True)
        bbob = Bbob(dimension)

        # Input vector with negative values
        input_vector = np.array([-3, -2, -1])
        rotation = np.array([
            [0.20076996547107095, 0.04246242617649648, 0.9787177138112657],
            [-0.4005763178669152, 0.9152789497892921, 0.042462426176496426],
            [-0.8939966636005579, -0.4005763178669152, 0.20076996547107095]
        ])
        result = rosenbrock.evaluate(input_vector, rotation)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = scaling_factor * \
            np.matmul(bbob.gram_schmidt(rotation.T), input_vector) + 1 / 2
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_dimension_1_rotated(self):
        dimension = 1
        rosenbrock = Rosenbrock(dimension, rotation=True)
        bbob = Bbob(dimension)

        # Evaluate at an arbitrary point
        input_vector = np.array([2.0])
        rotation = np.array([[-1]])
        result = rosenbrock.evaluate(input_vector, rotation)

        # Manually compute the expected result
        scaling_factor = max(1, rosenbrock.dimension / 8)
        z = scaling_factor * \
            np.matmul(bbob.gram_schmidt(rotation.T), input_vector) + 1 / 2
        expected_result = rosenbrock.raw(z) + rosenbrock.f_opt
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_evaluate_with_empty_input_vector_rotated(self):
        dimension = 3
        rosenbrock = Rosenbrock(dimension, rotation=True)

        # Empty input vector
        input_vector = np.array([])
        rotation = np.array([])
        with self.assertRaises(ValueError):
            rosenbrock.evaluate(input_vector, rotation)


if __name__ == '__main__':
    unittest.main()
