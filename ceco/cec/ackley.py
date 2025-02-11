from ceco.benchmark import Benchmark
import math


class Ackley(Benchmark):
    """
    A class representing the Ackley function, which is a benchmark function for optimization.

    The Ackley function is commonly used to test optimization algorithms due to its complex and multi-modal nature. The function is defined as:

    F(X) = -20 * exp(-0.2 * sqrt((1/D) * sum_{i=1}^D x_i^2)) - exp((1/D) * sum_{i=1}^D cos(2 * pi * x_i)) + 20 + e,

    where D is the dimensionality of the problem, and e is Euler's number (~2.71828).

    The class inherits from the `Benchmark` class and applies rotation and shift transformations to the input vector before evaluating the function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float], f_bias: float = 0) -> None:
        """
        Initializes the Ackley class with a rotation matrix and a shift vector.

        Ensures that the rotation matrix and shift vector have the same length. If their lengths do not match, a ValueError is raised.

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
        Evaluates the Ackley function at a given input vector after applying rotation and shift transformations.

        The function first applies the rotation matrix and shift vector to the input vector. Then, it computes the Ackley function:
        F(X) = -20 * exp(-0.2 * sqrt((1/D) * sum_{i=1}^D x_i^2)) - exp((1/D) * sum_{i=1}^D cos(2 * pi * x_i)) + 20 + e.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Ackley function after applying rotation and shift.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")

        # Apply shift and rotation
        shifted_rotated_vector = super().rotate_input(super().shift_input(input_vector))

        sum_term1 = sum(z_i ** 2 for z_i in shifted_rotated_vector)
        sum_term2 = sum(math.cos(2 * math.pi * z_i)
                        for z_i in shifted_rotated_vector)

        term1 = math.exp(-0.2 * math.sqrt(sum_term1 / dimension))
        term2 = -math.exp(sum_term2 / dimension)
        result = -20 * term1 + term2 + 20 + math.e + self.f_bias

        return result
