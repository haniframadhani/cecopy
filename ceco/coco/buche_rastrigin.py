import numpy as np
from ceco.bbob import Bbob


class Buche_rastrigin(Bbob):
    """
    A class representing the Buche-Rastrigin function, which is a benchmark function
    for optimization.

    The Buche-Rastrigin function is defined as:

        f(X)=10 ( D - sum(cos(2 pi x_i)) ) + sum(x_i ** 2) + 100 for i in [1, D]

    where x is an input vector of dimension D.

    This class inherits from `Bbob` and applies a shift transformation to the input vector. The optimal solution (x_opt) is randomly generated within the range [-5, 5], and the function value at x_opt (f_opt) computed as f(x_opt).

    Attributes:
        x_opt (np.ndarray): The optimal shift vector, randomly generated within [-5, 5].
    """

    def __init__(self, dimension: int) -> None:
        """
        Initializes the Buche-Rastrigin function with a given dimension.

        Parameters:
            dimension (int): The number of dimensions for the input space. Must be a positive integer.

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer")

        super().__init__(dimension)

        # Generate random optimal solution x_opt in range [-5, 5]
        self.x_opt = np.random.uniform(-5, 5, dimension)

        self.f_opt = self.raw(self.x_opt)

    def raw(self, x: np.ndarray) -> float:
        """
        Evaluates the Buche-Rastrigin function at a given input vector without any shift.

        The function is calculated as:

        f(X)=10 ( D - sum(cos(2 pi x_i)) ) + sum(x_i ** 2) + 100 for i in [1, D]

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
        sum_square = np.sum(x ** 2)
        result = 10 * (self.dimension - sum_cos) + sum_square + 100
        return result

    def compute_s_i(self, z_i: np.ndarray) -> np.ndarray:
        """
        Computes the scaling factors s_i for the input vector.

        The scaling factors are defined as:
            s_i = 10 * (10^(0.5 * (i - 1) / (D - 1)) if z_i > 0 and i is odd,
            s_i = 10^(0.5 * (i - 1) / (D - 1) otherwise.

        Parameters:
            z_i (np.ndarray): A vector of real numbers.

        Returns:
            np.ndarray: The scaling factors for the input vector.

        Notes:
            - If the dimension is 1, the scaling factor is always 1.
        """
        i = np.arange(1, self.dimension + 1)

        if self.dimension == 1:
            return np.ones_like(z_i)

        s_i = np.where(
            (z_i > 0) & (i % 2 == 1),
            10 * (10 ** (0.5 * (i - 1) / (self.dimension - 1))),
            10 ** (0.5 * (i - 1) / (self.dimension - 1))
        )
        return s_i

    def evaluate(self, input_vector: np.ndarray) -> float:
        """
        Evaluates the Buche-Rastrigin function at a given input vector.

        The function is calculated as:

        f(X)=10 ( D - sum(cos(2 pi z_i)) ) + sum(x_i ** 2) + 100 * f_pen(X) + f_opt

        where z = s_i * T_osz(x - x_opt).
        where s_i = 10 times 10^(0.5 times (i - 1) / (D - 1)),   if z_i > 0 and i is odd (i = 1, 3, 5, ...)
                10^(0.5 times (i - 1) / (D - 1)),        otherwise

        Parameters:
            input_vector (np.ndarray): A vector of real numbers representing a candidate solution. Must have the same length as the dimension of the Buche-Rastrigin function.

        Returns:
            float: The function value at the given input vector.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        if input_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Input vector must have {self.dimension} elements")

        # Shift the input vector
        z = input_vector - self.x_opt

        # Apply T_osz transformation
        z = self.T_osz(z)

        # Compute scaling factors s_i
        s = self.compute_s_i(z)

        # Element-wise multiplication
        z = self.elementwise_multiply(z, s)

        # Compute final function value
        result = self.raw(z) * self.f_pen(input_vector) + self.f_opt

        return result
