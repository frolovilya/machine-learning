import numpy as np
import pandas as pd


def normalize_variable(Xj):
    """
    @param Xj: (m x 1) vector,
               m - experiments count, 1 - one variable
    @return: normalized (m x 1) vector
    """
    return (Xj.values - np.mean(Xj.values)) / np.std(Xj.values)


def concat_with_x0(x):
    """
    @param x: (m x (n - 1)) DataFrame,
              n - variables count, m - experiments count
    @return: (m x n) DataFrame, with first column filled with 1
    """
    return pd.concat([pd.DataFrame([1] * x.shape[0]), x], axis=1)


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
