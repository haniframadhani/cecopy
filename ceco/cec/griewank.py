from ceco.benchmark import Benchmark
import math


class Griewank(Benchmark):
    """
    A class representing the Griewank function, a commonly used benchmark function
    in optimization problems. It inherits from the Benchmark class and applies
    rotation and shift transformations before evaluation.

    The Griewank function is defined as:

        f(x) = (1/4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i+1))) + 1

    where:
        - The first term represents the sum of squared components divided by 4000.
        - The second term is the product of cosine terms applied to each component.
        - The final term is a constant used to adjust the function's range.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming
            the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Griewank function with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix applied to the input vector.
            shift (list of float): The shift vector applied to the rotated input vector.

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
        Evaluates the rotated and shifted Griewank function at a given input vector.

        The function first applies a rotation transformation followed by a shift
        transformation to the input vector before computing the Griewank function.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The computed Griewank function value after applying rotation and shift.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")
        # Apply rotation
        rotated_vector = [0.0] * dimension
        for i in range(dimension):
            for j in range(dimension):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i]
                          for i in range(dimension)]

        sum_term = sum(z_i**2 / 4000 for z_i in shifted_vector)

        product_term = 1.0
        for i in range(dimension):
            product_term *= math.cos(shifted_vector[i]/math.sqrt(i+1))

        result = sum_term - product_term + 1
        return result
