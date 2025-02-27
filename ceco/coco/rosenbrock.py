from ceco.bbob import Bbob
import random
import math


class Rosenbrock(Bbob):
    """
    A class representing the Rosenbrock function, which is a benchmark function
    for optimization.

    The Rosenbrock function is defined as:

        f(X)=sum_{i=1}^{D-1}( 100( z_i^2-z_{i+1} )^2+( z_i-1 )^2 )+f_opt

    where x is an input vector of dimension D.

    This class inherits from `Bbob` and applies a shift transformation to the input vector. The optimal solution (x_opt) is randomly generated within the range [-5, 5], and the function value at x_opt (f_opt) can be explicitly provided or computed as f(x_opt).

    Attributes:
        x_opt (list of float): The optimal shift vector, randomly generated within [-4, 4].
        f_opt (float): The function value at x_opt. Defaults to f(x_opt) if not provided.
    """

    def __init__(self, dimension: int, f_opt: float = None) -> None:
        """
        Initializes the Rosenbrock function with a given dimension.

        Parameters:
            dimension (int): The number of dimensions for the input space. Must be a positive integer.
            f_opt (float, optional): The function value at x_opt. If None, it is computed as f(x_opt).

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        super().__init__(dimension)
        # Generate a random optimal solution vector x_opt within the range [-4, 4]
        self.x_opt = [random.uniform(-5, 5) for _ in range(dimension)]
        # Compute f_opt as the value of the Rosenbrock function at x_opt
        if f_opt is None:
            self.f_opt = self.raw(self.x_opt)
        else:
            self.f_opt = f_opt

    def raw(self, x: list[float]) -> float:
        """
        Evaluates the Rosenbrock function at a given input vector without any shift.

        The function is calculated as:

        f(X)=sum_{i=1}^{D-1}( 100( x_i^2-x_i+1 )^2+( x_i-1 )^2 )

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

        result = 0
        for i in range(self.dimension - 1):
            result += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        return result

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Rosenbrock function at a given input vector.

        The function is calculated as:

        f(X)=sum_{i=1}^{D-1}( 100( z_i^2-z_{i+1} )^2+( z_i-1 )^2 )+f_opt

        Parameters:
            input_vector (list[float]): A vector of real numbers representing a candidate solution. Must have the same length as the dimension of the Rosenbrock function.

        Returns:
            float: The function value at the given input vector.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        if len(input_vector) != self.dimension:
            raise ValueError(
                f"Input vector must have {self.dimension} elements")
        scaling_factor = max(1, self.dimension / 8)
        z = [scaling_factor * (input_vector[i]-self.x_opt[i]) +
             1 for i in range(self.dimension)]
        result = self.raw(z) + self.f_opt

        return result
