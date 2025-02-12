from ceco.benchmark import Benchmark
import math


class Different_power(Benchmark):
    """
    A class representing the Different Power benchmark function, used for optimization problems.

    The Different Power function is a unimodal, non-separable, and scalable function.
    It is defined as:
    f(X) = sqrt{sum_{i=1}^{D} left| x_i right|^{2 + 4 frac{i-1}{D-1}}}
    where X = [x_1, x_2, ..., x_D] is the input vector of dimension ( D ).

    This class inherits from the `Benchmark` class and implements the `evaluate` method
    to compute the value of the Discus function after applying shift and rotation
    transformations to the input vector.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float], f_bias: float = 0) -> None:
        """
        Initializes the Different_power class with a rotation matrix, a shift vector, and a bias term.

        Parameters:
            rotation (list of list of float): The rotation matrix for transforming the input vector.
            shift (list of float): The shift vector for adjusting the input vector.
            f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.

        Raises:
            ValueError: If `rotation` is not a non-empty square matrix.
            ValueError: If `shift` length does not match the dimension of `rotation`.
        """
        if not isinstance(rotation, list) or not rotation:
            raise ValueError(
                "Rotation matrix must be a non-empty list of lists")
        row_count = len(rotation)  # Number of rows
        if not all(isinstance(row, list) and len(row) == row_count for row in rotation):
            raise ValueError("Rotation matrix must be a square matrix")
        if len(rotation) != len(shift):
            raise ValueError("rotation and shift has different dimensions")
        super().__init__(rotation, shift, f_bias)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Different power function for the given input vector after applying shift and rotation transformations.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Different power function after applying rotation and shift.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")

        # Apply shift and rotation
        shifted_rotated_vector = super().rotate_input(super().shift_input(input_vector))

        total_sum = 0.0

        for i in range(1, dimension+1):
            exponent = 2 + 4 * ((i - 1) / (dimension - 1))
            total_sum += abs(shifted_rotated_vector[i-1])**exponent

        total_sum = math.sqrt(total_sum) + self.f_bias

        return total_sum
