from ceco.benchmark import Benchmark
import math


class Weierstrass(Benchmark):
    """
    A class representing the Weierstrass function, a continuous but nowhere differentiable function 
    used as a benchmark in optimization problems.

    Inherits from the Benchmark class and applies rotation and shift transformations to the 
    standard Weierstrass function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
        a (float): The decay factor, default is 0.5.
        b (float): The frequency factor, default is 3.
        k_max (int): The maximum number of summation terms, default is 20.
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float], f_bias: float = 0) -> None:
        """
        Initializes the Weierstrass class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix.
            shift (list of float): The shift vector.
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
        self.a = 0.5
        self.b = 3
        self.k_max = 20
        super().__init__(rotation, shift, f_bias)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the rotated and shifted Weierstrass function at a given input vector.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Weierstrass function after applying rotation and shift.

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
        # Calculate the first part of the function
        for i in range(dimension):
            inner_sum = 0.0
            for k in range(self.k_max + 1):
                inner_sum += self.a ** k * \
                    math.cos(2 * math.pi * self.b **
                             k * (shifted_rotated_vector[i] + 0.5))
            total_sum += inner_sum

        # Calculate the second part of the function
        second_sum = 0.0
        for k in range(self.k_max + 1):
            second_sum += self.a ** k * \
                math.cos(2 * math.pi * self.b ** k * 0.5)
        # Final result
        result = total_sum - dimension * second_sum + self.f_bias
        return result
