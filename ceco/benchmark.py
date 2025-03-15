import numpy as np


class Benchmark:
    """
    A base class for benchmark functions used in optimization.

    Attributes:
        dimension (int): The number of dimensions for the input space. Must be a positive integer.
        rotation (np.ndarray): A rotation matrix for transforming the input vector.
        shift (np.ndarray): A shift vector for adjusting the input vector.
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initializes the Benchmark class with a rotation matrix and a shift vector.

        Parameters:
            dimension (int): The number of dimensions for the input space. Must be a positive integer.

        Raises:
            ValueError: If `dimension` is not a positive integer.
        """
        if not isinstance(dimension, int):
            raise ValueError("dimension must be integer")
        if dimension < 0:
            raise ValueError("dimension must be positive integer")
        if dimension < 1:
            raise ValueError("dimension cannot be zero")
        self.dimension = dimension

    def cec_init(self, rotation: np.ndarray, shift: np.ndarray, f_bias: float = 0) -> None:
        """
        Initializes the cec function class with a rotation matrix and a shift vector.

        Parameters:
            rotation (np.ndarray): The rotation matrix.
            shift (np.ndarray): The shift vector.
            f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
        """
        if not isinstance(rotation, np.ndarray) or rotation.ndim != 2:
            raise ValueError(
                "Rotation matrix must be a non-empty np.ndarray")
        if rotation.shape[0] != self.dimension and shift.shape[0] != self.dimension:
            raise ValueError(
                "rotation dimension and shift dimension does not match dimension")
        elif rotation.shape[0] != self.dimension:
            raise ValueError(
                "rotation dimension does not match dimension")
        elif shift.shape[0] != self.dimension:
            raise ValueError(
                "shift dimension does not match dimension")
        if rotation.shape[0] != rotation.shape[1]:
            raise ValueError("Rotation matrix must be a square matrix")
        if rotation.shape[0] != shift.shape[0]:
            raise ValueError("rotation and shift has different dimensions")
        self.rotation = rotation
        self.shift = shift
        if not isinstance(f_bias, (float, int)):
            raise ValueError(
                "f_bias must be float or int")
        self.f_bias = float(f_bias)
