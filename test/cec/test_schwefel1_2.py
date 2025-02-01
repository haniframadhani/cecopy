import unittest
from ceco.cec.schwefel1_2 import Schwefel1_2


class Test_schwefel1_2(unittest.TestCase):
    def test_initialization(self):
        """Test initialization of the Schwefel1_2 class."""
        rotation = [[1.0, 0.0], [0.0, 1.0]]  # Identity matrix
        shift = [1.0, 2.0]

        schwefel = Schwefel1_2(rotation, shift)

        # Verify that rotation and shift are correctly set
        self.assertEqual(schwefel.rotation, rotation)
        self.assertEqual(schwefel.shift, shift)

    def test_evaluate_identity_rotation_and_zero_shift(self):
        """Test the Schwefel 1.2 function with identity rotation and zero shift."""
        rotation = [[1.0, 0.0], [0.0, 1.0]]  # Identity matrix
        shift = [0.0, 0.0]  # Zero shift
        schwefel = Schwefel1_2(rotation, shift)

        # Expected result: (1)^2 + (1+2)^2 = 1 + 9 = 10
        input_vector = [1.0, 2.0]
        result = schwefel.evaluate(input_vector)

        self.assertAlmostEqual(result, 10.0)

    def test_evaluate_with_rotation_and_shift(self):
        """Test the Schwefel 1.2 function with a custom rotation and shift."""
        rotation = [[0.0, 1.0], [1.0, 0.0]]  # Swaps x and y
        shift = [1.0, 2.0]
        schwefel = Schwefel1_2(rotation, shift)

        # Rotated vector: [2.0, 1.0], shifted: [1.0, -1.0]
        input_vector = [1.0, 2.0]
        # Expected result: (1)^2 + (1 + (-1))^2 = 1 + 0 = 1
        result = schwefel.evaluate(input_vector)

        self.assertAlmostEqual(result, 1.0)

    def test_evaluate_zero_vector(self):
        """Test the Schwefel 1.2 function with a zero input vector."""
        rotation = [[1.0, 0.0], [0.0, 1.0]]  # Identity matrix
        shift = [1.0, 2.0]
        schwefel = Schwefel1_2(rotation, shift)

        input_vector = [0.0, 0.0]  # Shifted vector: [-1.0, -2.0]
        # Expected result: (-1)^2 + (-1 + (-2))^2 = 1 + 9 = 10
        result = schwefel.evaluate(input_vector)

        self.assertAlmostEqual(result, 10.0)


if __name__ == "__main__":
    unittest.main()
