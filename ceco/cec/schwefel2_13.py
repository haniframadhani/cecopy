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

    The Schwefel 2.13 function is defined as:
    F(X) = sum_{i=1 to D} (A_i - B_i(X))^2,
    where:
    - A_i = sum_{j=1 to D} (a_{ij} * sin(alpha_j) + b_{ij} * cos(alpha_j)),
    - B_i(X) = sum_{j=1 to D} (a_{ij} * sin(x_j) + b_{ij} * cos(x_j)).

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
        dimension (int): The dimensionality of the problem.
        a (list of list of int): A D x D matrix of random integers in the range [-100, 100].
        b (list of list of int): A D x D matrix of random integers in the range [-100, 100].
        alpha (list of float): A vector of D random numbers in the range [-π, π].
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float], dimension: int, f_bias: float = 0) -> None:
        """
        Initializes the Schwefel2_13 class with a rotation matrix, shift vector, and dimension.

        During initialization, random matrices `a` and `b` and a random vector `alpha` are
        generated. These are used to define the Schwefel 2.13 function.

        Parameters:
            rotation (list of list of float): The rotation matrix for transforming the input vector.
            shift (list of float): The shift vector for adjusting the input vector.
            dimension (int): The dimensionality of the problem.
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
        self.dimension = dimension
        if len(rotation) != dimension or len(shift) != dimension:
            raise ValueError("rotation and shift has different dimensions")
        super().__init__(rotation, shift, f_bias)

        # Generate random matrices a and b
        self.a = self.generate_random_matrix(self.dimension, -100, 100)
        self.b = self.generate_random_matrix(self.dimension, -100, 100)

        # Generate random alpha vector
        self.alpha = self.generate_random_alpha(self.dimension)

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

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")

        # Apply shift and rotation
        shifted_rotated_vector = super().rotate_input(super().shift_input(input_vector))

        # Compute A and B
        A = self.compute_A(self.dimension, self.a, self.b, self.alpha)
        B = self.compute_B(self.dimension, self.a,
                           self.b, shifted_rotated_vector)

        # Compute the Schwefel 2.13 function
        total_sum = 0.0
        for i in range(self.dimension):
            total_sum += (A[i] - B[i]) ** 2

        return total_sum + self.f_bias
