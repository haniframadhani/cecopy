import unittest
from ceco.cec.sphere import Sphere


class Test_sphere(unittest.TestCase):
    def test_initialization(self):
        """Test initialization of the Sphere class."""
        rotation = [[1.0, 0.0], [0.0, 1.0]]
        shift = [1.0, 2.0]

        sphere = Sphere(rotation, shift)

        self.assertEqual(sphere.rotation, rotation)
        self.assertEqual(sphere.shift, shift)

    def test_evaluate_identity_rotation_and_zero_shift(self):
        """Test the Sphere function with identity rotation and zero shift."""
        rotation = [[1.0, 0.0], [0.0, 1.0]]
        shift = [0.0, 0.0]
        sphere = Sphere(rotation, shift)

        input_vector = [3.0, 4.0]  # Expected result is 3^2 + 4^2 = 25
        result = sphere.evaluate(input_vector)

        self.assertAlmostEqual(result, 25.0)

    def test_evaluate_with_rotation_and_shift(self):
        """Test the Sphere function with a custom rotation and shift."""
        rotation = [[0.0, 1.0], [1.0, 0.0]]  # Swaps x and y
        shift = [1.0, 2.0]
        sphere = Sphere(rotation, shift)

        # Rotated vector: [4.0, 3.0], shifted: [3.0, 1.0]
        input_vector = [3.0, 4.0]
        # Result: 3^2 + 1^2 = 10
        result = sphere.evaluate(input_vector)

        self.assertAlmostEqual(result, 10.0)

    def test_evaluate_zero_vector(self):
        """Test the Sphere function with a zero input vector."""
        rotation = [[1.0, 0.0], [0.0, 1.0]]
        shift = [1.0, 2.0]
        sphere = Sphere(rotation, shift)

        input_vector = [0.0, 0.0]  # Shifted vector: [-1.0, -2.0]
        # Result: (-1)^2 + (-2)^2 = 5
        result = sphere.evaluate(input_vector)

        self.assertAlmostEqual(result, 5.0)


if __name__ == "__main__":
    unittest.main()
