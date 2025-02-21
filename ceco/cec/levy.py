from ceco.benchmark import Benchmark
import math


class Levy(Benchmark):
    """
    A class representing the Levy benchmark function, used for optimization problems.

    It is defined as:
    f(X)=sin^2(pi w_1)+ sum_{i=1}^{D-1}(w_i-1)^2[ 1+10 sin^2(pi w_i+1) ]+(w_D-1)^2[ 1+ sin^2(2 pi w_D) ]

    where X = [x_1, x_2, ..., x_D] is the input vector of dimension ( D ).

    This class inherits from the `Benchmark` class and implements the `evaluate` method to compute the value of the Discus function after applying shift and rotation transformations to the input vector.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float], f_bias: float = 0) -> None:
        """
        Initializes the Levy class with a rotation matrix, a shift vector, and a bias term.

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
        Evaluates the Levy function for the given input vector after applying shift and rotation transformations.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Levy function after applying rotation and shift.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")

        # Apply shift and rotation
        shifted_rotated_vector = super().rotate_input(super().shift_input(input_vector))

        w = [1 + (x - 1) / 4.0 for x in shifted_rotated_vector]

        term1 = math.sin(math.pi * w[0]) ** 2

        sum_term = 0.0
        for i in range(dimension-1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 *
                                           math.sin(math.pi * w[i] + 1) ** 2)
        term2 = (w[-1] - 1) ** 2 * \
            (1 + math.sin(2 * math.pi * w[-1]) ** 2)
        result = term1 + sum_term + term2 + self.f_bias
        return result
