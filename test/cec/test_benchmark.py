import unittest
from ceco.benchmark import Benchmark


class Test_benchmark(unittest.TestCase):
    def setUp(self):
        # Example rotation matrix and shift vector for testing
        # Identity matrix (no rotation)
        self.input_2d = [3, 2]
        self.input_3d = [3, 2, 1]
        self.no_rotation_2d = [[1, 0], [0, 1]]
        self.no_rotation_3d = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.rotation_2d = [[0, -1], [1, 0]]  # 90-degree rotation
        self.rotation_3d = [[0, -1, 0], [1, 0, 0],
                            [0, 0, 1]]  # 90-degree rotation
        self.shift_2d = [1, 1]  # Shift vector
        self.no_shift_2d = [0, 0]  # No shift vector
        self.shift_3d = [1, 1, 1]  # Shift vector
        self.no_shift_3d = [0, 0, 0]  # No shift vector

    def test_rotate_input_no_rotation_2d(self):
        benchmark = Benchmark(self.no_rotation_2d, self.no_shift_2d)
        result = benchmark.rotate_input(self.input_2d)
        expected_result = [3, 2]
        self.assertListEqual(result, expected_result)

    def test_rotate_input_rotation_2d(self):
        benchmark = Benchmark(self.rotation_2d, self.no_shift_2d)
        result = benchmark.rotate_input(self.input_2d)
        expected_result = [2, -3]
        self.assertListEqual(result, expected_result)

    def test_rotate_input_no_rotation_3d(self):
        benchmark = Benchmark(self.no_rotation_3d, self.no_shift_3d)
        result = benchmark.rotate_input(self.input_3d)
        expected_result = [3, 2, 1]
        self.assertListEqual(result, expected_result)

    def test_rotate_input_rotation_3d(self):
        benchmark = Benchmark(self.rotation_3d, self.no_shift_3d)
        result = benchmark.rotate_input(self.input_3d)
        expected_result = [2, -3, 1]
        self.assertListEqual(result, expected_result)

    def test_shift_input_no_shift_2d(self):
        benchmark = Benchmark(self.rotation_2d, self.no_shift_2d)
        result = benchmark.shift_input(self.input_2d)
        expected_result = [3, 2]
        self.assertListEqual(result, expected_result)

    def test_shift_input_shift_2d(self):
        benchmark = Benchmark(self.rotation_2d, self.shift_2d)
        result = benchmark.shift_input(self.input_2d)
        expected_result = [2, 1]
        self.assertListEqual(result, expected_result)

    def test_shift_input_no_shift_3d(self):
        benchmark = Benchmark(self.rotation_3d, self.no_shift_3d)
        result = benchmark.shift_input(self.input_3d)
        expected_result = [3, 2, 1]
        self.assertListEqual(result, expected_result)

    def test_shift_input_shift_3d(self):
        benchmark = Benchmark(self.rotation_3d, self.shift_3d)
        result = benchmark.shift_input(self.input_3d)
        expected_result = [2, 1, 0]
        self.assertListEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
