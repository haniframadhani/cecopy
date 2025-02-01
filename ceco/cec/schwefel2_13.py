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
        D (int): The dimensionality of the problem, inferred from the length of the shift vector.
        a (list of list of int): A D x D matrix of random integers in the range [-100, 100].
        b (list of list of int): A D x D matrix of random integers in the range [-100, 100].
        alpha (list of float): A vector of D random numbers in the range [-π, π].
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Schwefel2_13 class with a rotation matrix and a shift vector.

        During initialization, random matrices `a` and `b` and a random vector `alpha` are
        generated. These are used to define the Schwefel 2.13 function.

        Parameters:
            rotation (list of list of float): The rotation matrix for transforming the input vector.
            shift (list of float): The shift vector for adjusting the input vector.
        """
        super().__init__(rotation, shift)

        # Generate random matrices a and b
        self.D = len(shift)  # Dimension of the problem
        self.a = self.generate_random_matrix(self.D, -100, 100)
        self.b = self.generate_random_matrix(self.D, -100, 100)

        # Generate random alpha vector
        self.alpha = self.generate_random_alpha(self.D)

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

    def compute_A(self) -> list[float]:
        """
        Computes the A vector using the formula:
        A_i = sum_{j=1 to D} (a_{ij} * sin(alpha_j) + b_{ij} * cos(alpha_j)).

        The A vector is a fixed vector computed using the random matrices `a` and `b`
        and the random vector `alpha`.

        Returns:
            list of float: The computed A vector.
        """
        A = [0.0] * self.D
        for i in range(self.D):
            for j in range(self.D):
                A[i] += self.a[i][j] * \
                    math.sin(self.alpha[j]) + \
                    self.b[i][j] * math.cos(self.alpha[j])
        return A

    def compute_B(self, X: list[float]) -> list[float]:
        """
        Computes the B vector using the formula:
        B_i(X) = sum_{j=1 to D} (a_{ij} * sin(x_j) + b_{ij} * cos(x_j)).

        The B vector is computed for a given input vector X using the random matrices `a` and `b`.

        Parameters:
            X (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            list of float: The computed B vector.
        """
        B = [0.0] * self.D
        for i in range(self.D):
            for j in range(self.D):
                B[i] += self.a[i][j] * \
                    math.sin(X[j]) + self.b[i][j] * math.cos(X[j])
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
        # Apply rotation
        rotated_vector = [0.0] * len(input_vector)
        for i in range(len(input_vector)):
            for j in range(len(input_vector)):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i]
                          for i in range(len(rotated_vector))]

        # Compute A and B
        A = self.compute_A()
        B = self.compute_B(shifted_vector)

        # Compute the Schwefel 2.13 function
        total_sum = 0.0
        for i in range(self.D):
            total_sum += (A[i] - B[i]) ** 2

        return total_sum
