from ceco.benchmark import Benchmark


class Ellipsoid(Benchmark):
    """
    A class representing the Ellipsoid function, which is a benchmark function
    for optimization.

    The Ellipsoid function is defined as:

    F(X) = sum_{i=1}^{D} i * (x_i)^2

    where:
    - D is the dimension of the input vector.
    - x = (x1, x2, ..., xD) is a D-dimensional row vector (i.e., a 1xD matrix).
    - The function is characterized by its elongated shape, which can make optimization algorithms struggle to find the global minimum.

    This class inherits from the Benchmark class and applies rotation and shift transformations to the input vector before evaluating the function.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Ellipsoid class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix for transforming the input vector.
            shift (list of float): The shift vector for adjusting the input vector.

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
        if len(rotation) != len(shift):
            raise ValueError("rotation and shift has different dimensions")
        super().__init__(rotation, shift)

    def evaluate(self, input_vector: list[float]) -> float:
        """
        Evaluates the Ellipsoid function at a given input vector after applying
        rotation and shift transformations.

        The evaluation process involves the following steps:
        1. Apply the rotation matrix to the input vector to obtain the rotated vector.
        2. Apply the shift vector to the rotated vector to obtain the shifted vector.
        3. Calculate the Ellipsoid function using the shifted vector.

        The Ellipsoid function is computed as follows:

        F(X) = sum_{i=1}^{D} i * (x_i - shift_i)^2

        where:
        - x_i is the i-th element of the shifted vector.

        Parameters:
            input_vector (list of float): The input vector [x1, x2, ..., xD].

        Returns:
            float: The result of the Ellipsoid function after applying rotation and shift.

        Raises:
            ValueError: If the input vector does not match the expected dimension.
        """
        dimension = len(input_vector)
        if len(self.rotation) != dimension and len(self.shift) != dimension:
            raise ValueError(
                "Input vector dimension does not match rotation and shift dimensions")
        rotated_vector = [0.0] * dimension

        for i in range(dimension):
            for j in range(dimension):
                rotated_vector[i] += self.rotation[i][j] * input_vector[j]

        # Apply shift
        shifted_vector = [rotated_vector[i] - self.shift[i]
                          for i in range(dimension)]

        # Calculate the Ellipsoid function
        total_sum = 0.0
        for i in range(1, dimension + 1):
            total_sum += i * (shifted_vector[i - 1] ** 2)

        return total_sum
