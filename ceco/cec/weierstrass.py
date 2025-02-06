from ceco.benchmark import Benchmark
import math


class Weierstrass(Benchmark):
    """
    A class representing the Weierstrass function, a continuous but nowhere differentiable function 
    used as a benchmark in optimization problems.

    Inherits from the Benchmark class and applies rotation and shift transformations to the 
    standard Weierstrass function.

    Attributes:
        a (float): The decay factor, default is 0.5.
        b (float): The frequency factor, default is 3.
        k_max (int): The maximum number of summation terms, default is 20.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Weierstrass class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix.
            shift (list of float): The shift vector.
        """
        self.a = 0.5
        self.b = 3
        self.k_max = 20
        super().__init__(rotation, shift)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the rotated and shifted Weierstrass function at a given input vector.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Weierstrass function after applying rotation and shift.
        """
        dimension = len(input_vector)
        # Apply rotation
        rotated_vector = [0.0] * dimension
        for i in range(dimension):
            for j in range(dimension):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i]
                          for i in range(dimension)]
        total_sum = 0.0
        # Calculate the first part of the function
        for i in range(dimension):
            inner_sum = 0.0
            for k in range(self.k_max + 1):
                inner_sum += self.a ** k * \
                    math.cos(2 * math.pi * self.b **
                             k * (shifted_vector[i] + 0.5))
            total_sum += inner_sum

        # Calculate the second part of the function
        second_sum = 0.0
        for k in range(self.k_max + 1):
            second_sum += self.a ** k * \
                math.cos(2 * math.pi * self.b ** k * 0.5)
        # Final result
        result = total_sum - dimension * second_sum
        return result
