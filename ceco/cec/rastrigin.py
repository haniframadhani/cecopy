from ceco.benchmark import Benchmark
import math


class Rastrigin(Benchmark):
    """
    A class representing the Rastrigin function, which is a benchmark function
    for optimization.

    The Rastrigin function is defined as:
    F(X) = sum_{i=1}^{D} (x_i^2 - 10 * cos(2 * pi * x_i) + 10)

    The class inherits from the `Benchmark` class and applies rotation and shift
    transformations to the input vector before evaluating the function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Rastrigin class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix for transforming the input vector.
            shift (list of float): The shift vector for adjusting the input vector.
        """
        super().__init__(rotation, shift)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Rastrigin function at a given input vector after applying
        rotation and shift transformations.

        The function first applies the rotation matrix and shift vector to the input vector.
        Then, it computes the Rastrigin function:
        F(X) = sum_{i=1}^{D} (x_i^2 - 10 * cos(2 * pi * x_i) + 10).

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Rastrigin function after applying rotation and shift.
        """
        # Infer dimension from the input vector
        dimension = len(input_vector)

        # Apply rotation
        rotated_vector = [0.0] * dimension
        for i in range(dimension):
            for j in range(dimension):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i]
                          for i in range(dimension)]

        # Compute the Rastrigin function
        total_sum = sum(
            z_i**2 - 10 * math.cos(2 * math.pi * z_i) + 10 for z_i in shifted_vector
        )

        return total_sum
