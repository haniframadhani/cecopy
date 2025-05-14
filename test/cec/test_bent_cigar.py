import unittest
from ceco.cec.bent_cigar import Bent_cigar
import numpy as np


class Test_bent_cigar(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = np.array([[1, 0], [0, 1]])
        self.rotation_non_identity = np.array(
            [[0, -1], [1, 0]])  # 90-degree rotation
        self.shift = np.array([1, 1])  # Shift vector
        self.no_shift = np.array([0, 0])  # No shift vector
        self.f_bias = 1.0
        self.dimension = 2

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        bent_cigar = Bent_cigar(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])  # Example input
        result = bent_cigar.evaluate(input_vector)
        expected_result = 9_000_004
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        bent_cigar = Bent_cigar(
            self.dimension, self.rotation_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = bent_cigar.evaluate(input_vector)
        expected_result = (2-1)**2+10**6*((3-1)**2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        bent_cigar = Bent_cigar(
            self.dimension, self.rotation_non_identity, self.no_shift)
        input_vector = np.array([2.0, 3.0])
        result = bent_cigar.evaluate(input_vector)
        expected_result = 4_000_009
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        bent_cigar = Bent_cigar(
            self.dimension, self.rotation_non_identity, self.shift)
        input_vector = np.array([2.0, 3.0])
        result = bent_cigar.evaluate(input_vector)
        expected_result = 1_000_004
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        bent_cigar = Bent_cigar(
            self.dimension, self.rotation_identity, self.no_shift)
        input_vector = np.array([0.0, 0.0])
        result = bent_cigar.evaluate(input_vector)
        expected_result = 0.0
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_f_bias(self):
        bent_cigar = Bent_cigar(
            self.dimension, self.rotation_identity, self.no_shift, self.f_bias)
        input_vector = np.array([0.0, 0.0])
        result = bent_cigar.evaluate(input_vector)
        expected_result = 1.0
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
