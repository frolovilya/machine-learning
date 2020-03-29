import numpy as np


def normalize_variable(x):
    """
    :param x: (m x n) vector,
               m - experiments count, n - variables
    :return: normalized (m x n) vector, mu - column means, sigma - column std deviations
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x - mu, axis=0)

    return [np.divide(x - mu, sigma), mu, sigma]


def concat_with_x0(x):
    """
    Concat x with x0 (ones) vector

    :param x: (m x n) matrix
    :return: (m x (n + 1)) matrix with first column containing all ones
    """
    return np.hstack((np.ones((x.shape[0], 1)), x))


def roll_vector_to_list_of_matrices(unrolled_vector, shapes):
    """
    Roll vector back to list of matrices

    :param unrolled_vector: unrolled vector
    :param shapes: list of k original shapes
    :return: k matrices
    """
    first = 0
    result = []

    for shape in shapes:
        last = first + shape[0] * shape[1]
        result.append(np.array(unrolled_vector[first:last]).reshape(shape[0], shape[1]))
        first = last

    return result


def unroll_list_of_matrices_to_vector(matrices_list):
    """
    Unroll matrices to single vector

    :param matrices_list: list of k matrices
    :return: (list of k original matrix shapes, unrolled vector)
    """
    result_vector = np.array([])
    original_shape = []

    for m in matrices_list:
        original_shape.append(m.shape)
        result_vector = np.hstack([
            result_vector,
            m.reshape(m.shape[0] * m.shape[1])
        ])

    return original_shape, result_vector
