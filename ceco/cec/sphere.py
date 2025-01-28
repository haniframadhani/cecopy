from ceco.benchmark import Benchmark


class Sphere(Benchmark):
    """
    A class representing the Sphere function, which is a benchmark function
    for optimization.

    Inherits from the Benchmark class and applies rotation and shift to the
    standard sphere function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming
            the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Sphere class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix.
            shift (list of float): The shift vector.
        """
        super().__init__(rotation, shift)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the rotated and shifted sphere function at a given input vector.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the sphere function after applying rotation
            and shift.
        """
        # Apply rotation
        rotated_vector = [0] * len(input_vector)
        for i in range(len(input_vector)):
            for j in range(len(input_vector)):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i]
                          for i in range(len(rotated_vector))]

        # Calculate the sum of squares
        total_sum_of_squares = 0
        for value in shifted_vector:
            total_sum_of_squares += value ** 2

        return total_sum_of_squares
