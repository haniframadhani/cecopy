from ceco.bbob import Bbob
import random
import math


class Buche_Rastrigin(Bbob):
    """
    A class representing the Buche-Rastrigin function, which is a benchmark function
    for optimization.

    The Buche-Rastrigin function is defined as:

        f(X)=10 ( D - sum(cos(2 pi x_i)) ) + sum(x_i ** 2) + 100 for i in [1, D]

    where x is an input vector of dimension D.

    This class inherits from `Bbob` and applies a shift transformation to the input vector. The optimal solution (x_opt) is randomly generated within the range [-5, 5], and the function value at x_opt (f_opt) can be explicitly provided or computed as f(x_opt).

    Attributes:
        x_opt (list of float): The optimal shift vector, randomly generated within [-5, 5].
        f_opt (float): The function value at x_opt. Defaults to f(x_opt) if not provided.
    """

    def __init__(self, dimension: int, f_opt: float = None) -> None:
        """
        Initializes the Buche-Rastrigin function with a given dimension.

        Parameters:
            dimension (int): The number of dimensions for the input space. Must be a positive integer.
            f_opt (float, optional): The function value at x_opt. If None, it is computed as f(x_opt).

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        super().__init__(dimension)
        # Generate a random optimal solution vector x_opt within the range [-5, 5]
        self.x_opt = [random.uniform(-5, 5) for _ in range(dimension)]
        # Compute f_opt as the value of the Buche-Rastrigin function at x_opt
        if f_opt is None:
            self.f_opt = self.raw(self.x_opt)
        else:
            self.f_opt = f_opt

    def raw(self, x: list[float]) -> float:
        """
        Evaluates the Buche-Rastrigin function at a given input vector without any shift.

        The function is calculated as:

        f(X)=10 ( D - sum(cos(2 pi x_i)) ) + sum(x_i ** 2) + 100 for i in [1, D]

        Parameters:
            x (list[float]): A vector of real numbers representing a candidate solution.

        Returns:
            float: The function value at the given input vector.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        if len(x) != self.dimension:
            raise ValueError(
                f"Input vector must have {self.dimension} elements")

        sum_cos = sum(math.cos(2 * math.pi * x_i) for x_i in x)
        sum_square = sum(x_i ** 2 for x_i in x)
        result = 10 * (self.dimension - sum_cos) + sum_square + 100
        return result

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Buche-Rastrigin function at a given input vector.

        The function is calculated as:

        f(X)=10 ( D - sum(cos(2 pi z_i)) ) + sum(x_i ** 2) + 100 * f_pen(X) + f_opt

        where z = s_i * T_osz(x - x_opt).
        where s_i = 10 times 10^(0.5 times (i - 1) / (D - 1)),   if z_i > 0 and i is odd (i = 1, 3, 5, ...)
                10^(0.5 times (i - 1) / (D - 1)),        otherwise

        Parameters:
            input_vector (list[float]): A vector of real numbers representing a candidate solution. Must have the same length as the dimension of the Buche-Rastrigin function.

        Returns:
            float: The function value at the given input vector.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        if len(input_vector) != self.dimension:
            raise ValueError(
                f"Input vector must have {self.dimension} elements")
        z = [x - y for x, y in zip(input_vector, self.x_opt)]
        z = self.T_osz(z)
        s = self.compute_s_i(z)
        z = [x * y for x, y in zip(z, s)]
        result = self.raw(z) * self.f_pen(input_vector) + self.f_opt

        return result
