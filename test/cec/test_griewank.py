import unittest
import math
from ceco.cec.griewank import Griewank


class Test_griewank(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        griewank = Griewank(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]
        result = griewank.evaluate(input_vector)
        expected_result = (2**2/4000)+(3**2/4000) - \
            (math.cos(2/math.sqrt(1))*math.cos(3/math.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        griewank = Griewank(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]
        result = griewank.evaluate(input_vector)
        expected_result = ((-3)**2/4000)+(2**2/4000) - \
            (math.cos((-3)/math.sqrt(1))*math.cos(2/math.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        griewank = Griewank(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = griewank.evaluate(input_vector)
        expected_result = ((1)**2/4000)+(2**2/4000) - \
            (math.cos((1)/math.sqrt(1))*math.cos(2/math.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        griewank = Griewank(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]
        result = griewank.evaluate(input_vector)
        expected_result = ((-4)**2/4000)+(1**2/4000) - \
            (math.cos((-4)/math.sqrt(1))*math.cos(1/math.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        griewank = Griewank(self.rotation_identity, self.no_shift)
        input_vector = [0.0, 0.0]
        result = griewank.evaluate(input_vector)
        expected_result = (0**2/4000)+(0**2/4000) - \
            (math.cos(0/math.sqrt(1))*math.cos(0/math.sqrt(2)))+1
        self.assertAlmostEqual(result, expected_result, places=5)
