import unittest
import numpy as np
from ceco.benchmark import Benchmark


class Test_benchmark(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.input_2d = np.array([3, 2])
        self.input_3d = np.array([3, 2, 1])
        self.no_rotation_2d = np.array([[1, 0], [0, 1]])
        self.no_rotation_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.rotation_2d = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        self.rotation_3d = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90-degree rotation
        self.shift_2d = np.array([1, 1])  # Shift vector
        self.no_shift_2d = np.array([0, 0])  # No shift vector
        self.shift_3d = np.array([1, 1, 1])  # Shift vector
        self.no_shift_3d = np.array([0, 0, 0])  # No shift vector
        self.f_bias = 100

    def test_initialization_2d_no_rotation_no_shift(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.no_rotation_2d, self.no_shift_2d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.no_rotation_2d, benchmark.rotation)
        np.testing.assert_array_equal(self.no_shift_2d, benchmark.shift)

    def test_initialization_3d_no_rotation_no_shift(self):
        dimension = 3
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.no_rotation_3d, self.no_shift_3d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.no_rotation_3d, benchmark.rotation)
        np.testing.assert_array_equal(self.no_shift_3d, benchmark.shift)

    def test_initialization_2d_rotation_no_shift(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.rotation_2d, self.no_shift_2d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.rotation_2d, benchmark.rotation)
        np.testing.assert_array_equal(self.no_shift_2d, benchmark.shift)

    def test_initialization_3d_rotation_no_shift(self):
        dimension = 3
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.rotation_3d, self.no_shift_3d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.rotation_3d, benchmark.rotation)
        np.testing.assert_array_equal(self.no_shift_3d, benchmark.shift)

    def test_initialization_2d_rotation_shift(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.rotation_2d, self.shift_2d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.rotation_2d, benchmark.rotation)
        np.testing.assert_array_equal(self.shift_2d, benchmark.shift)

    def test_initialization_3d_rotation_shift(self):
        dimension = 3
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.rotation_3d, self.shift_3d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.rotation_3d, benchmark.rotation)
        np.testing.assert_array_equal(self.shift_3d, benchmark.shift)

    def test_initialization_2d_no_rotation_shift(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.no_rotation_2d, self.shift_2d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.no_rotation_2d, benchmark.rotation)
        np.testing.assert_array_equal(self.shift_2d, benchmark.shift)

    def test_initialization_3d_no_rotation_shift(self):
        dimension = 3
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.no_rotation_3d, self.shift_3d)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.no_rotation_3d, benchmark.rotation)
        np.testing.assert_array_equal(self.shift_3d, benchmark.shift)

    def test_mismatch_dimension_rotation_shift(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        with self.assertRaises(ValueError) as context:
            benchmark.cec_init(self.no_rotation_3d, self.no_shift_3d)
        self.assertEqual(str(context.exception),
                         "rotation dimension and shift dimension does not match dimension")

    def test_mismatch_dimension_rotation(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        with self.assertRaises(ValueError) as context:
            benchmark.cec_init(self.no_rotation_3d, self.no_shift_2d)
        self.assertEqual(str(context.exception),
                         "rotation dimension does not match dimension")

    def test_mismatch_dimension_shift(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        with self.assertRaises(ValueError) as context:
            benchmark.cec_init(self.no_rotation_2d, self.no_shift_3d)
        self.assertEqual(str(context.exception),
                         "shift dimension does not match dimension")

    def test_negative_dimension(self):
        dimension = -1
        with self.assertRaises(ValueError) as context:
            benchmark = Benchmark(dimension)
        self.assertEqual(str(context.exception),
                         "dimension must be positive integer")

    def test_zero_dimension(self):
        dimension = 0
        with self.assertRaises(ValueError) as context:
            benchmark = Benchmark(dimension)
        self.assertEqual(str(context.exception),
                         "dimension cannot be zero")

    def test_non_int_dimension(self):
        float = 1.0
        string = "abc"
        li = [1, 2, 3]
        with self.assertRaises(ValueError) as context:
            benchmark = Benchmark(float)
            self.assertEqual(str(context.exception),
                             "dimension must be integer")
            benchmark = Benchmark(string)
            self.assertEqual(str(context.exception),
                             "dimension must be integer")
            benchmark = Benchmark(li)
            self.assertEqual(str(context.exception),
                             "dimension must be integer")

    def test_initialization_f_bias(self):
        dimension = 2
        benchmark = Benchmark(dimension)
        benchmark.cec_init(self.rotation_2d, self.shift_2d, self.f_bias)
        self.assertEqual(dimension, benchmark.dimension)
        np.testing.assert_array_equal(self.rotation_2d, benchmark.rotation)
        np.testing.assert_array_equal(self.shift_2d, benchmark.shift)
        self.assertEqual(self.f_bias, benchmark.f_bias)

    def test_not_int_not_float_f_bias(self):
        dimension = 2
        string = "abc"
        li = [1, 2, 3]
        benchmark = Benchmark(dimension)
        with self.assertRaises(ValueError) as context:
            benchmark.cec_init(self.no_rotation_2d, self.no_shift_2d, string)
            self.assertEqual(str(context.exception),
                             "f_bias must be float or int")
        with self.assertRaises(ValueError) as context:
            benchmark.cec_init(self.no_rotation_2d, self.no_shift_2d, li)
            self.assertEqual(str(context.exception),
                             "f_bias must be float or int")


if __name__ == '__main__':
    unittest.main()
