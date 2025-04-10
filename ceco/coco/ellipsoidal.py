import numpy as np
from ceco.benchmark import Benchmark


class Ellipsoidal(Benchmark):
    """
    A class representing the Ellipsoidal function, which is a benchmark function
    for optimization.

    The Ellipsoidal function is defined as:

        f(x) = sum(10^(6 * (i-1)/(D-1) * (x_i)^2) for i in [1, D]

    where x is an input vector of dimension D.

    This class inherits from `Benchmark` and applies a shift transformation to the input vector. The optimal solution (x_opt) is randomly generated within the range [-5, 5], and the function value at x_opt (f_opt) computed as f(x_opt).

    Attributes:
        x_opt (np.ndarray): The optimal shift vector, randomly generated within [-5, 5].
        f_opt (float): The function value at x_opt.
        high_conditioning (bool): If `True`, applies an additional Gram-Schmidt transformation to introduce high conditioning.
    """

    def __init__(self, dimension: int, high_conditioning: bool = None) -> None:
        """
        Initializes the Ellipsoidal function with a given dimension.

        Parameters:
            dimension (int): The number of dimensions for the input space. Must be a positive integer.
            high_conditioning (bool, optional): If `True`, applies a Gram-Schmidt transformation to increase problem conditioning. Defaults to `False`.

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Dimension must be a positive integer")
        super().__init__(dimension)
        # Generate a random optimal solution vector x_opt within the range [-5, 5]
        self.x_opt = np.random.uniform(-5, 5, dimension)

        self.f_opt = self.raw(self.x_opt)

        self.high_conditioning = high_conditioning

    def raw(self, x: np.ndarray) -> float:
        """
        Evaluates the Ellipsoidal function at a given input vector without any shift.

        The function is calculated as:
            f(x) = sum(10^(6 * (i-1)/(D-1) * (x_i)^2) for i in [1, D]

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

        i = np.arange(1, self.dimension + 1)
        if self.dimension == 1:
            exponent = 0  # Handle the case when dimension = 1
        else:
            exponent = 6 * (i - 1) / (self.dimension - 1)
        total_sum = np.sum((10 ** exponent) * (x ** 2))

        return total_sum

    def evaluate(self, input_vector: np.ndarray) -> float:
        """
        Evaluates the Ellipsoidal function at a given input vector.

        The function is calculated as:
            f(x) = sum(10^(6 * (i-1)/(D-1) * (z_i)^2) + f_opt

        Parameters:
            input_vector (np.ndarray): A vector of real numbers representing a candidate solution. Must have the same length as the dimension of the Ellipsoidal function.

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
        if not self.high_conditioning:
            z = self.T_osz(z)
        else:
            z = self.T_osz(self.gram_schmidt(np.array([z]).T).ravel())

        # Compute the raw Ellipsoidal function value
        result = self.raw(z) + self.f_opt

        return result
