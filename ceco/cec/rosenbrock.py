from ceco.benchmark import Benchmark


class Rosenbrock(Benchmark):
    """
    A class representing the Rosenbrock function, which is a benchmark function
    for optimization.

    The Rosenbrock function is defined as:
    F(X) = sum_{i=1}^{D-1} [100 * (x_i^2 - x_{i+1})^2 + (x_i - 1)^2]

    The class inherits from the `Benchmark` class and applies rotation and shift
    transformations to the input vector before evaluating the function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Rosenbrock class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix for transforming the input vector.
            shift (list of float): The shift vector for adjusting the input vector.

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
        Evaluates the Rosenbrock function at a given input vector after applying
        rotation and shift transformations.

        The function first applies the rotation matrix and shift vector to the input vector.
        Then, it computes the Rosenbrock function:
        F(X) = sum_{i=1}^{D-1} [100 * (x_i^2 - x_{i+1})^2 + (x_i - 1)^2].

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Rosenbrock function after applying rotation and shift.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")

        # Apply shift and rotation
        shifted_rotated_vector = super().rotate_input(super().shift_input(input_vector))
        # Calculate the Rosenbrock function
        total_sum = 0.0

        for i in range(dimension - 1):
            term1 = 100 * \
                (shifted_rotated_vector[i] ** 2 -
                 shifted_rotated_vector[i + 1]) ** 2
            term2 = (shifted_rotated_vector[i] - 1) ** 2
            total_sum += term1 + term2
        return total_sum
