import numpy as np
from ceco.benchmark import Benchmark


class Rastrigin(Benchmark):
    """
    A class representing the Rastrigin function, which is a benchmark function
    for optimization.

    The Rastrigin function is defined as:

        f(X)=10 ( D - sum(cos(2 pi x_i)) ) for i in [1, D]

    where x is an input vector of dimension D.

    This class inherits from `Benchmark` and applies a shift transformation to the input vector. The optimal solution (x_opt) is randomly generated within the range [-5, 5], and the function value at x_opt (f_opt) computed as f(x_opt).

    Attributes:
        x_opt (np.ndarray): The optimal shift vector, randomly generated within [-5, 5].
        f_opt (float): The function value at x_opt.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initializes the Rastrigin function with a given dimension.

        Parameters:
            dimension (int): The number of dimensions for the input space. Must be a positive integer.

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        super().__init__(dimension)
        # Generate a random optimal solution vector x_opt within the range [-5, 5]
        self.x_opt = np.random.uniform(-5, 5, dimension)

        self.f_opt = self.raw(self.x_opt)

    def raw(self, x: np.ndarray) -> float:
        """
        Evaluates the Rastrigin function at a given input vector without any shift.

        The function is calculated as:

        f(X)=10 ( D - sum(cos(2 pi x_i)) ) for i in [1, D]

        Parameters:
            x (np.ndarray): A vector of real numbers representing a candidate solution.

        Returns:
            float: The function value at the given input vector.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        if x.shape[0] != self.dimension:
            raise ValueError(
                f"Input vector must have {self.dimension} elements")

        sum_cos = np.sum(np.cos(2 * np.pi * x))
        result = 10 * (self.dimension - sum_cos)
        return result

    def evaluate(self, input_vector: np.ndarray) -> float:
        """
        Evaluates the Rastrigin function at a given input vector.

        The function is calculated as:

        f(X)=10 ( D - sum(cos(2 pi z_i)) ) + ||z||^2 + f_opt

        where z = Λ^10 * T_asy(0.2, T_osz(x - x_opt)).

        Parameters:
            input_vector (np.ndarray): A vector of real numbers representing a candidate solution. Must have the same length as the dimension of the Rastrigin function.

        Returns:
            float: The function value at the given input vector.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        if input_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Input vector must have {self.dimension} elements")
        z = input_vector - self.x_opt
        z = self.T_osz(z)
        z = np.matmul(
            self.create_diagonal_matrix(10), self.T_asy_beta(0.2, z))
        result = self.raw(z) + self.euclidean_norm(z) + self.f_opt

        return result
