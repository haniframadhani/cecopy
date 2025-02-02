import unittest
from ceco.cec.elliptic import Elliptic


class Test_elliptic(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.rotation_identity = [[1, 0], [0, 1]]
        self.rotation_non_identity = [[0, -1], [1, 0]]  # 90-degree rotation
        self.shift = [1, 1]  # Shift vector
        self.no_shift = [0, 0]  # No shift vector

    def test_evaluate_with_identity_rotation_and_no_shift(self):
        elliptic = Elliptic(self.rotation_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        expected_result = (10**6) ** (0 / 1) * (2.0 ** 2) + \
            (10**6) ** (1 / 1) * (3.0 ** 2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_identity_rotation_and_shift(self):
        elliptic = Elliptic(self.rotation_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        expected_result = (10**6) ** (0 / 1) * (2.0 - 1) ** 2 + \
            (10**6) ** (1 / 1) * (3.0 - 1) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_no_shift(self):
        elliptic = Elliptic(self.rotation_non_identity, self.no_shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        # After rotation: [3.0, 2.0]
        expected_result = (10**6) ** (0 / 1) * (3.0 ** 2) + \
            (10**6) ** (1 / 1) * (2.0 ** 2)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_non_identity_rotation_and_shift(self):
        elliptic = Elliptic(self.rotation_non_identity, self.shift)
        input_vector = [2.0, 3.0]  # Example input
        result = elliptic.evaluate(input_vector)
        # After rotation: [-3.0, 2.0]
        expected_result = (10**6) ** (0 / 1) * (-3.0 - 1) ** 2 + \
            (10**6) ** (1 / 1) * (2.0 - 1) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_evaluate_with_zero_input(self):
        elliptic = Elliptic(self.rotation_identity, self.shift)
        input_vector = [0.0, 0.0]  # Known input
        result = elliptic.evaluate(input_vector)
        expected_result = (10**6) ** (0 / 1) * (0.0 - 1) ** 2 + \
            (10**6) ** (1 / 1) * (0.0 - 1) ** 2
        self.assertAlmostEqual(result, expected_result, places=5)


if __name__ == '__main__':
    unittest.main()
