class Benchmark:
    """
    A base class for benchmark functions used in optimization.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming
            the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
        f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float], f_bias: float = 0) -> None:
        """
        Initializes the Benchmark class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix.
            shift (list of float): The shift vector.
            f_bias (float): A bias term added to the benchmark function's output. Defaults to 0.
        """
        self.rotation = rotation
        self.shift = shift
        self.f_bias = f_bias

    def rotate_input(self, input_vector: list[float]) -> list[float]:
        """
        Multiplies a 1xD matrix input_vector with an DxD matrix rotation.

        Parameters:
            input_vector (list of float): The 1xD matrix (a list of D elements).

        Returns:
            list of float: The resulting 1xD matrix after multiplication.
        """
        dimension = len(input_vector)  # Number of columns in X and rows in Y

        # Initialize the result as a list of zeros with length N
        result = [0.0] * dimension

        # Perform matrix multiplication
        for i in range(dimension):  # Iterate over columns of Y
            for j in range(dimension):  # Iterate over rows of Y
                result[i] += input_vector[j] * self.rotation[j][i]

        return result

    def shift_input(self, input_vector: list[float]) -> list[float]:
        """
        Applies the shift transformation to the input vector.

        Parameters:
            input_vector (list of float): The input vector to be shifted.

        Returns:
            list of float: The resulting vector after applying the shift transformation.
        """
        return [input_vector[i] - self.shift[i] for i in range(len(input_vector))]
