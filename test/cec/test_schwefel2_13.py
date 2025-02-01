import unittest
from ceco.cec.schwefel2_13 import Schwefel2_13
import math


class Test_schwefel2_13(unittest.TestCase):
    def test_initialization(self):
        """
        Test initialization of the Schwefel2_13 class.
        """
        rotation = [[1.0, 0.0], [0.0, 1.0]]  # Identity matrix
        shift = [1.0, 2.0]

        schwefel = Schwefel2_13(rotation, shift)

        # Verify that rotation and shift are correctly set
        self.assertEqual(schwefel.rotation, rotation)
        self.assertEqual(schwefel.shift, shift)

        # Verify that random matrices a and b are generated
        self.assertEqual(len(schwefel.a), 2)  # D = 2
        self.assertEqual(len(schwefel.b), 2)  # D = 2

        # Verify that random alpha vector is generated
        self.assertEqual(len(schwefel.alpha), 2)  # D = 2

    def test_evaluate_identity_rotation_and_zero_shift(self):
        """
        Test the Schwefel 2.13 function with identity rotation and zero shift.
        """
        rotation = [[1.0, 0.0], [0.0, 1.0]]  # Identity matrix
        shift = [0.0, 0.0]  # Zero shift
        schwefel = Schwefel2_13(rotation, shift)

        # Set fixed matrices a and b for reproducibility
        schwefel.a = [[1, 2], [3, 4]]
        schwefel.b = [[5, 6], [7, 8]]
        # Fixed alpha for reproducibility
        schwefel.alpha = [math.pi / 4, math.pi / 6]

        input_vector = [1.0, 2.0]  # Input vector
        result = schwefel.evaluate(input_vector)

        # Manually compute expected result
        A = schwefel.compute_A()
        B = schwefel.compute_B(input_vector)  # No shift or rotation applied
        expected_result = sum((A[i] - B[i]) ** 2 for i in range(2))

        self.assertAlmostEqual(result, expected_result)

    def test_evaluate_with_rotation_and_shift(self):
        """
        Test the Schwefel 2.13 function with a custom rotation and shift.
        """
        rotation = [[0.0, 1.0], [1.0, 0.0]]  # Swaps x and y
        shift = [1.0, 2.0]
        schwefel = Schwefel2_13(rotation, shift)

        # Set fixed matrices a and b for reproducibility
        schwefel.a = [[1, 2], [3, 4]]
        schwefel.b = [[5, 6], [7, 8]]
        # Fixed alpha for reproducibility
        schwefel.alpha = [math.pi / 4, math.pi / 6]

        input_vector = [1.0, 2.0]  # Input vector
        result = schwefel.evaluate(input_vector)

        # Manually compute expected result
        rotated_vector = [rotation[0][0] * input_vector[0] + rotation[0][1] * input_vector[1],
                          rotation[1][0] * input_vector[0] + rotation[1][1] * input_vector[1]]
        shifted_vector = [rotated_vector[0] -
                          shift[0], rotated_vector[1] - shift[1]]
        A = schwefel.compute_A()
        B = schwefel.compute_B(shifted_vector)
        expected_result = sum((A[i] - B[i]) ** 2 for i in range(2))

        self.assertAlmostEqual(result, expected_result)

    def test_evaluate_zero_vector(self):
        """
        Test the Schwefel 2.13 function with a zero input vector.
        """
        rotation = [[1.0, 0.0], [0.0, 1.0]]  # Identity matrix
        shift = [1.0, 2.0]
        schwefel = Schwefel2_13(rotation, shift)

        # Set fixed matrices a and b for reproducibility
        schwefel.a = [[1, 2], [3, 4]]
        schwefel.b = [[5, 6], [7, 8]]
        # Fixed alpha for reproducibility
        schwefel.alpha = [math.pi / 4, math.pi / 6]

        input_vector = [0.0, 0.0]  # Zero input vector
        result = schwefel.evaluate(input_vector)

        # Manually compute expected result
        rotated_vector = [rotation[0][0] * input_vector[0] + rotation[0][1] * input_vector[1],
                          rotation[1][0] * input_vector[0] + rotation[1][1] * input_vector[1]]
        shifted_vector = [rotated_vector[0] -
                          shift[0], rotated_vector[1] - shift[1]]
        A = schwefel.compute_A()
        B = schwefel.compute_B(shifted_vector)
        expected_result = sum((A[i] - B[i]) ** 2 for i in range(2))

        self.assertAlmostEqual(result, expected_result)

    def test_evaluate_no_rotation_no_shift(self):
        """
        Test the Schwefel 2.13 function with no rotation and no shift.
        """
        rotation = [[1.0, 0.0], [0.0, 1.0]]  # Identity matrix
        shift = [0.0, 0.0]  # Zero shift
        schwefel = Schwefel2_13(rotation, shift)

        # Set fixed matrices a and b for reproducibility
        schwefel.a = [[1, 2], [3, 4]]
        schwefel.b = [[5, 6], [7, 8]]
        # Fixed alpha for reproducibility
        schwefel.alpha = [math.pi / 4, math.pi / 6]

        input_vector = [1.0, 2.0]  # Input vector
        result = schwefel.evaluate(input_vector)

        # Manually compute expected result
        A = schwefel.compute_A()
        B = schwefel.compute_B(input_vector)  # No shift or rotation applied
        expected_result = sum((A[i] - B[i]) ** 2 for i in range(2))

        self.assertAlmostEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
