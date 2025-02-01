from ceco.benchmark import Benchmark
import math
import random


class Schwefel2_13(Benchmark):
    """
    A class representing the Schwefel 2.13 function, which is a benchmark function
    for optimization.

    This function is commonly used to test optimization algorithms due to its complex
    and multi-modal nature. The class inherits from the `Benchmark` class and applies
    rotation and shift transformations to the input vector before evaluating the function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Schwefel2_13 class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix for transforming the input vector.
            shift (list of float): The shift vector for adjusting the input vector.
        """
        super().__init__(rotation, shift)

    def generate_random_matrix(self, D: int, min_val: int, max_val: int) -> list[list[int]]:
        """
        Generates a D x D matrix with random integers in the range [min_val, max_val].

        Parameters:
            D (int): The dimension of the matrix.
            min_val (int): The minimum value for random integers.
            max_val (int): The maximum value for random integers.

        Returns:
            list of list of int: A D x D matrix of random integers.
        """
        return [[random.randint(min_val, max_val) for _ in range(D)] for _ in range(D)]

    def generate_random_alpha(self, D: int) -> list[float]:
        """
        Generates a list of D random numbers in the range [-π, π].

        Parameters:
            D (int): The number of random numbers to generate.

        Returns:
            list of float: A list of D random numbers in the range [-π, π].
        """
        return [random.uniform(-math.pi, math.pi) for _ in range(D)]

    def compute_A(self, D: int, a: list[list[int]], b: list[list[int]], alpha: list[float]) -> list[float]:
        """
        Computes the A vector using the formula:
        A_i = sum_{j=1 to D} (a_{ij} * sin(alpha_j) + b_{ij} * cos(alpha_j)).

        Parameters:
            D (int): The dimension of the problem.
            a (list of list of int): The a matrix.
            b (list of list of int): The b matrix.
            alpha (list of float): The alpha vector.

        Returns:
            list of float: The computed A vector.
        """
        A = [0.0] * D
        for i in range(D):
            for j in range(D):
                A[i] += a[i][j] * math.sin(alpha[j]) + \
                    b[i][j] * math.cos(alpha[j])
        return A

    def compute_B(self, D: int, a: list[list[int]], b: list[list[int]], X: list[float]) -> list[float]:
        """
        Computes the B vector using the formula:
        B_i(X) = sum_{j=1 to D} (a_{ij} * sin(x_j) + b_{ij} * cos(x_j)).

        Parameters:
            D (int): The dimension of the problem.
            a (list of list of int): The a matrix.
            b (list of list of int): The b matrix.
            X (list of float): The input vector.

        Returns:
            list of float: The computed B vector.
        """
        B = [0.0] * D
        for i in range(D):
            for j in range(D):
                B[i] += a[i][j] * math.sin(X[j]) + b[i][j] * math.cos(X[j])
        return B

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Schwefel 2.13 function at a given input vector after applying
        rotation and shift transformations.

        The function first applies the rotation matrix and shift vector to the input vector.
        Then, it computes the A and B vectors and evaluates the Schwefel 2.13 function:
        F(X) = sum_{i=1 to D} (A_i - B_i(X))^2.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Schwefel 2.13 function after applying rotation and shift.
        """
        # Infer dimension from the input vector
        D = len(input_vector)

        # Generate random matrices a and b
        a = self.generate_random_matrix(D, -100, 100)
        b = self.generate_random_matrix(D, -100, 100)

        # Generate random alpha vector
        alpha = self.generate_random_alpha(D)

        # Apply rotation
        rotated_vector = [0.0] * D
        for i in range(D):
            for j in range(D):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i] for i in range(D)]

        # Compute A and B
        A = self.compute_A(D, a, b, alpha)
        B = self.compute_B(D, a, b, shifted_vector)

        # Compute the Schwefel 2.13 function
        total_sum = 0.0
        for i in range(D):
            total_sum += (A[i] - B[i]) ** 2

        return total_sum
