from ceco.benchmark import Benchmark


class Sphere(Benchmark):
    """
    A class representing the Sphere function, which is a benchmark function
    for optimization.

    Inherits from the Benchmark class and applies rotation and shift to the
    standard sphere function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Sphere class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix.
            shift (list of float): The shift vector.
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
        super().__init__(rotation, shift)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the rotated and shifted sphere function at a given input vector.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the sphere function after applying rotation
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

        # Calculate the sum of squares
        total_sum_of_squares = 0
        for value in shifted_rotated_vector:
            total_sum_of_squares += value ** 2

        return total_sum_of_squares
