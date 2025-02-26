import math
import random
from typing import List, Union


class Bbob:
    """
    A base class for Bbob functions used in optimization.
    """

    def __init__(self, dimension: int) -> None:
        """
        Initializes the Bbob class with the dimension of the input vector.

        Parameters:
            dimension (int): The dimension of the input vector.
        """
        self.dimension = dimension

    def euclidean_norm(self, input_vector: List[float]) -> float:
        """
        Computes the Euclidean norm (L2 norm) of the input vector.

        Parameters:
            input_vector (List[float]): The input vector.

        Returns:
            float: The Euclidean norm of the input vector.

        Raises:
            ValueError: If the input vector is empty.
        """
        if not input_vector:
            raise ValueError("Input vector cannot be empty.")
        return math.sqrt(sum(x_i ** 2 for x_i in input_vector))

    def create_diagonal_matrix(self, alpha: float) -> List[List[float]]:
        """
        Creates a diagonal matrix Λ^alpha with diagonal elements defined as:
        λ_ii = alpha^((1/2) * (i-1)/(D-1)), for i = 1, ..., D.

        Parameters:
            alpha (float): Base value for the diagonal elements.

        Returns:
            List[List[float]]: The diagonal matrix as a list of lists.
        """
        if self.dimension == 1:
            return [[alpha ** ((1 / 2) * 0)]]  # Handle the case when D = 1

        # Handle the case when alpha = 0
        if alpha == 0:
            return [[0.0] * self.dimension for _ in range(self.dimension)]

        # Compute the diagonal elements
        diagonal_elements = [alpha ** ((1 / 2) * (i - 1) / (self.dimension - 1))
                             for i in range(1, self.dimension + 1)]

        # Create the diagonal matrix as a list of lists
        matrix = [[0.0] * self.dimension for _ in range(self.dimension)]
        for i in range(self.dimension):
            matrix[i][i] = diagonal_elements[i]  # Set the diagonal elements

        return matrix

    def dot_product(self, v1: List[float], v2: List[float]) -> float:
        """
        Computes the dot product of two vectors.

        Parameters:
            v1 (List[float]): The first vector.
            v2 (List[float]): The second vector.

        Returns:
            float: The dot product of v1 and v2.
        """
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length.")
        return sum(x * y for x, y in zip(v1, v2))

    def vector_norm(self, v: List[float]) -> float:
        """
        Computes the Euclidean norm (magnitude) of a vector.

        Parameters:
            v (List[float]): The input vector.

        Returns:
            float: The Euclidean norm of the vector.
        """
        return math.sqrt(self.dot_product(v, v))

    def vector_subtract(self, v1: List[float], v2: List[float]) -> List[float]:
        """
        Subtracts two vectors element-wise.

        Parameters:
            v1 (List[float]): The first vector.
            v2 (List[float]): The second vector.

        Returns:
            List[float]: The resulting vector after subtraction.
        """
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length.")
        return [x - y for x, y in zip(v1, v2)]

    def vector_scale(self, v: List[float], scalar: float) -> List[float]:
        """
        Scales a vector by a scalar.

        Parameters:
            v (List[float]): The input vector.
            scalar (float): The scalar to multiply the vector by.

        Returns:
            List[float]: The scaled vector.
        """
        return [x * scalar for x in v]

    def gram_schmidt(self, matrix: List[List[float]]) -> List[List[float]]:
        """
        Applies the Gram-Schmidt process to a matrix to make it orthogonal.

        Parameters:
            matrix (List[List[float]]): The input matrix with standard normally distributed entries.

        Returns:
            List[List[float]]: The orthogonal matrix.
        """
        if not matrix or len(matrix) != len(matrix[0]):
            raise ValueError("Matrix must be square and non-empty.")

        # Convert the matrix to a list of vectors (rows)
        vectors = [list(row) for row in matrix]
        orthogonal_vectors = []

        for v in vectors:
            # Subtract the projection of v onto each orthogonal vector in orthogonal_vectors
            for u in orthogonal_vectors:
                projection = self.vector_scale(
                    u, self.dot_product(v, u) / self.dot_product(u, u))
                v = self.vector_subtract(v, projection)
            # Normalize the resulting vector
            norm = self.vector_norm(v)
            if norm > 0:  # Avoid division by zero
                v = self.vector_scale(v, 1 / norm)
            orthogonal_vectors.append(v)

        # Transpose the result to match the input format (columns are orthogonal vectors)
        orthogonal_matrix = list(map(list, zip(*orthogonal_vectors)))
        return orthogonal_matrix

    def generate_random_matrix(self, D: int) -> List[List[float]]:
        """
        Generates a random matrix with standard normally distributed entries.

        Parameters:
            D (int): The dimension of the matrix.

        Returns:
            List[List[float]]: The generated random matrix.
        """
        return [[random.gauss(0, 1) for _ in range(D)] for _ in range(D)]

    def matrix_transpose(self, matrix: List[List[float]]) -> List[List[float]]:
        """
        Computes the transpose of a matrix.

        Parameters:
            matrix (List[List[float]]): The input matrix.

        Returns:
            List[List[float]]: The transposed matrix.
        """
        return list(map(list, zip(*matrix)))

    def matrix_multiply(self, A: Union[List[List[float]], List[float]],
                        B: Union[List[List[float]], List[float]]) -> Union[List[List[float]], List[float], float]:
        """
        Multiplies two matrices or a matrix with a vector, supporting 1D and 2D inputs.
        Also supports dot product when both A and B are 1D vectors.

        Parameters:
            A (List[List[float]] or List[float]): The first matrix or vector.
            B (List[List[float]] or List[float]): The second matrix or vector.

        Returns:
            List[List[float]] or List[float] or float: The product of A and B.
        """
        # Check if A is a 1D vector
        is_A_vector = isinstance(A[0], (int, float))
        if is_A_vector:
            A = [A]  # Convert row vector to a single-row matrix

        # Check if B is a 1D vector and convert it to a column matrix if necessary
        is_B_vector = isinstance(B[0], (int, float))
        if is_B_vector:
            B = [[b] for b in B]  # Convert vector to column matrix

        # Validate dimensions
        if len(A[0]) != len(B):
            raise ValueError(
                "Incompatible dimensions: Cannot multiply A and B.")

        # Perform matrix multiplication
        result = [[sum(a * b for a, b in zip(row_A, col_B)) for col_B in zip(*B)]
                  for row_A in A]

        # If both A and B were originally 1D vectors, return a scalar (dot product result)
        if is_A_vector and is_B_vector:
            return result[0][0]  # Return a single number instead of a list

        # If B was originally a vector, return a flattened list
        return [row[0] for row in result] if is_B_vector else result

    def T_asy_beta(self, beta: float, input_vector: List[float]) -> List[float]:
        """
        Applies the transformation T_asy^beta to the input vector.

        Parameters:
            beta (float): The transformation parameter.
            input_vector (List[float]): The input vector.

        Returns:
            List[float]: The transformed vector.
        """
        if self.dimension == 1:
            return input_vector  # No transformation needed for D = 1

        y = []
        for i in range(self.dimension):
            x_i = input_vector[i]
            if x_i > 0:
                exponent = 1 + beta * \
                    (i) / (self.dimension - 1) * math.sqrt(x_i)
                y_i = x_i ** exponent
            else:
                y_i = x_i
            y.append(y_i)
        return y

    def T_osz(self, input_vector: List[float]) -> List[float]:
        """
        Applies the transformation T_osz to the input vector.

        Parameters:
            input_vector (List[float]): The input vector.

        Returns:
            List[float]: The transformed vector.
        """
        transformed_vector = []
        for x in input_vector:
            if x == 0:
                x_hat = 0
            else:
                x_hat = math.log(abs(x))

            if x > 0:
                c1, c2 = 10, 7.9
            else:
                c1, c2 = 5.5, 3.1

            sine_term = 0.049 * (math.sin(c1 * x_hat) + math.sin(c2 * x_hat))
            transformed = math.exp(x_hat + sine_term)

            if x < 0:
                transformed_vector.append(-transformed)
            elif x == 0:
                transformed_vector.append(0)
            else:
                transformed_vector.append(transformed)
        return transformed_vector

    def elementwise_multiply(self, x: List[float], y: List[float]) -> List[float]:
        """
        Computes the element-wise multiplication of two vectors.

        Parameters:
            x (List[float]): The first vector.
            y (List[float]): The second vector.

        Returns:
            List[float]: The element-wise product of x and y.
        """
        if len(x) != len(y):
            raise ValueError("Vectors must have the same length.")
        return [xi * yi for xi, yi in zip(x, y)]

    def f_pen(self, x: list[float]) -> float:
        """
        Computes the penalty function f_pen(X) = sum(max(0, |x_i| - 5)^2 for i in [1, D].

        Parameters:
            X (list[float]): A list of real numbers representing the input vector.

        Returns:
            float: The penalty value.
        """
        penalty = 0.0
        for x_i in x:
            penalty += max(0, abs(x_i) - 5) ** 2
        return penalty

    def compute_s_i(self, z_i: list[float]) -> list[float]:
        """
        Computes the scaling factors s_i for a given list of transformed variables z_i.

        s_i = 10 times 10^(0.5 times (i - 1) / (D - 1)),   if z_i > 0 and i is odd (i = 1, 3, 5, ...)
                10^(0.5 times (i - 1) / (D - 1)),        otherwise

        Parameters:
            z_i (list[float]): A list of transformed variables.

        Returns:
            list[float]: A list of scaling factors s_i corresponding to each z_i.
        """
        s_i = []

        for i, z in enumerate(z_i, start=1):  # Start indexing from 1
            if self.dimension == 1:  # Handle the case when D = 1
                s_i.append(1.0)  # Default scaling factor for D = 1
            else:
                if z > 0 and i % 2 == 1:  # Odd index and z_i > 0
                    s_i.append(
                        10 * (10 ** (0.5 * (i - 1) / (self.dimension - 1))))
                else:
                    s_i.append(10 ** (0.5 * (i - 1) / (self.dimension - 1)))

        return s_i

# # Example input vector
# input_vector = [1.0, 2.0, 3.0]

# # Create an instance of the Bbob class
# bbob = Bbob(dimension=3)

# # Compute the Euclidean norm
# norm = bbob.euclidean_norm(input_vector)
# print("Euclidean norm:", norm)

# # Create a diagonal matrix
# alpha = 2.0
# a = 1.0
# b = 1.0
# diagonal_matrix = bbob.create_diagonal_matrix(alpha, a, b)
# print("Diagonal matrix:")
# for row in diagonal_matrix:
#     print(row)

# # Apply the T_asy_beta transformation
# beta = 1.0
# transformed_vector = bbob.T_asy_beta(beta, input_vector)
# print("Transformed vector (T_asy_beta):", transformed_vector)

# # Apply the T_osz transformation
# transformed_vector = bbob.T_osz(input_vector)
# print("Transformed vector (T_osz):", transformed_vector)

# # Element-wise multiplication
# x = [1, 2, 3]
# y = [4, 5, 6]
# result = bbob.elementwise_multiply(x, y)
# print("Element-wise multiplication:", result)
