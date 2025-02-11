from ceco.benchmark import Benchmark


class Schwefel1_2(Benchmark):
    """
    A class representing the Schwefel 1.2 function, which is a benchmark function
    for optimization.

    Inherits from the Benchmark class and applies rotation and shift to the
    input vector before evaluating the Schwefel 1.2 function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming
            the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float], f_bias: float = 0) -> None:
        """
        Initializes the Schwefel1_2 class with a rotation matrix and a shift vector.

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
        super().__init__(rotation, shift, f_bias)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Schwefel 1.2 function at a given input vector after applying
        rotation and shift.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Schwefel 1.2 function after applying rotation
            and shift.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")

        # Apply shift and rotation
        shifted_rotated_vector = super().rotate_input(super().shift_input(input_vector))

        # Compute the Schwefel 1.2 function
        total_sum = 0.0

        for i in range(1, dimension + 1):
            cumulative_sum = sum(shifted_rotated_vector[:i])
            total_sum += cumulative_sum ** 2

        return total_sum + self.f_bias
