import numpy as np


class Bbob:
    """
    A base class for implementing BBOB (Black-Box Optimization Benchmark) functions using NumPy.

    This class provides common utility methods and transformations used in BBOB benchmark functions,
    such as Euclidean norm, diagonal matrix creation, random matrix generation, and various transformations.

    Attributes:
        dimension (int): The dimensionality of the input space for the benchmark function.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initializes the Bbob class with a given dimension.

        Parameters:
            dimension (int): The number of dimensions for the input space. Must be a positive integer.

        Raises:
            ValueError: If `dimension` is not a positive integer.
        """
        self.dimension = dimension

    def euclidean_norm(self, input_vector: np.ndarray) -> float:
        """
        Computes the Euclidean norm (L2 norm) of a given input vector.

        Parameters:
            input_vector (np.ndarray): A vector of real numbers.

        Returns:
            float: The Euclidean norm of the input vector.

        Raises:
            ValueError: If the input vector is empty.
        """
        if input_vector.size == 0:
            raise ValueError("Input vector cannot be empty.")
        return np.linalg.norm(input_vector)

    def create_diagonal_matrix(self, alpha: float) -> np.ndarray:
        """
        Creates a diagonal matrix Λ^α, where the diagonal elements are powers of α.

        The diagonal elements are defined as:
            Λ_ii = alpha^(0.5 * (i - 1) / (D - 1)) for i = 1, 2, ..., D,
            where D is the dimension.

        Parameters:
            alpha (float): The base value for the diagonal elements.

        Returns:
            np.ndarray: A diagonal matrix of shape (D, D), where D is the dimension.

        Notes:
            - If the dimension is 1, the matrix is [[1]].
            - If alpha is 0, the matrix is a zero matrix.
        """
        if self.dimension == 1:
            return np.array([[alpha ** 0]])

        if alpha == 0:
            return np.zeros((self.dimension, self.dimension))

        diagonal = np.array([alpha ** (0.5 * (i - 1) / (self.dimension - 1))
                             for i in range(1, self.dimension + 1)])
        return np.diag(diagonal)

    def generate_random_matrix(self, D: int) -> np.ndarray:
        """
        Generates a random matrix with elements drawn from a standard normal distribution.

        Parameters:
            D (int): The size of the matrix (D x D).

        Returns:
            np.ndarray: A random matrix of shape (D, D).
        """
        return np.random.randn(D, D)

    def matrix_transpose(self, matrix: np.ndarray) -> np.ndarray:
        """
        Computes the transpose of a given matrix.

        Parameters:
            matrix (np.ndarray): A matrix of shape (M, N).

        Returns:
            np.ndarray: The transpose of the matrix, of shape (N, M).
        """
        return matrix.T

    def T_asy_beta(self, beta: float, input_vector: np.ndarray) -> np.ndarray:
        """
        Applies the T_asy^beta transformation to a given input vector.

        The transformation is defined as:
            T_asy^beta(x_i) = x_i^(1 + beta * (i / (D - 1)) * sqrt(x_i)) if x_i > 0,
            T_asy^beta(x_i) = x_i otherwise.

        Parameters:
            beta (float): The beta parameter controlling the transformation.
            input_vector (np.ndarray): A vector of real numbers.

        Returns:
            np.ndarray: The transformed vector.

        Notes:
            - If the dimension is 1, the input vector is returned unchanged.
        """
        if self.dimension == 1:
            return input_vector

        i = np.arange(self.dimension)
        mask = input_vector > 0
        exponent = 1 + beta * (i / (self.dimension - 1)
                               ) * np.sqrt(input_vector)
        result = np.where(mask, input_vector ** exponent, input_vector)
        return result

    def T_osz(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Applies the T_osz transformation to a given input vector.

        The transformation is defined as:
            T_osz(x_i) = sign(x_i) * exp(x_hat + 0.049 * (sin(c1 * x_hat) + sin(c2 * x_hat))),
            where x_hat = log(|x_i|) if x_i ≠ 0, and 0 otherwise.

        Parameters:
            input_vector (np.ndarray): A vector of real numbers.

        Returns:
            np.ndarray: The transformed vector.

        Notes:
            - The transformation is applied element-wise.
            - If x_i is 0, the result is 0.
        """
        mask_zero = input_vector == 0
        mask_positive = input_vector > 0

        x_hat = np.where(mask_zero, 0, np.log(np.abs(input_vector)))

        c1 = np.where(mask_positive, 10, 5.5)
        c2 = np.where(mask_positive, 7.9, 3.1)

        sine_term = 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat))
        transformed = np.exp(x_hat + sine_term)

        result = np.where(mask_positive, transformed, -transformed)

        # Set 0 where input was 0
        result[mask_zero] = 0

        return result

    def elementwise_multiply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Performs element-wise multiplication of two vectors or matrices.

        Parameters:
            x (np.ndarray): The first input array.
            y (np.ndarray): The second input array.

        Returns:
            np.ndarray: The element-wise product of x and y.

        Raises:
            ValueError: If the shapes of x and y do not match.
        """
        if x.shape != y.shape:
            raise ValueError("Vectors must have the same length.")
        return x * y

    def f_pen(self, x: np.ndarray) -> float:
        """
        Computes the penalty function for a given input vector.

        The penalty function is defined as:
            f_pen(x) = sum(max(0, |x_i| - 5)^2).

        Parameters:
            x (np.ndarray): A vector of real numbers.

        Returns:
            float: The penalty value.
        """
        return np.sum(np.maximum(0, np.abs(x) - 5) ** 2)
