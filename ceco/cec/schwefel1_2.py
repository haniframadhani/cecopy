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
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Schwefel1_2 class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix.
            shift (list of float): The shift vector.
        """
        super().__init__(rotation, shift)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Schwefel 1.2 function at a given input vector after applying
        rotation and shift.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Schwefel 1.2 function after applying rotation
            and shift.
        """
        # Apply rotation
        rotated_vector = [0.0] * len(input_vector)
        for i in range(len(input_vector)):
            for j in range(len(input_vector)):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i]
                          for i in range(len(rotated_vector))]

        # Compute the Schwefel 1.2 function
        dimentions = len(shifted_vector)
        total_sum = 0.0

        for i in range(1, dimentions + 1):
            cumulative_sum = sum(shifted_vector[:i])
            total_sum += cumulative_sum ** 2

        return total_sum
