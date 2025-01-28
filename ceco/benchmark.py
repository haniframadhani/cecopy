class Benchmark:
    """
    A base class for benchmark functions used in optimization.

    Attributes:
        rotation (list of list of float): A rotation matrix for transforming
            the input vector.
        shift (list of float): A shift vector for adjusting the input vector.
    """

    def __init__(self, rotation: list[list[float]], shift: list[float]) -> None:
        """
        Initializes the Benchmark class with a rotation matrix and a shift vector.

        Parameters:
            rotation (list of list of float): The rotation matrix.
            shift (list of float): The shift vector.
        """
        self.rotation = rotation
        self.shift = shift
