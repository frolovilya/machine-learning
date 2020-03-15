import numpy as np
from utils import *


def cost_function(coefficients, x, y, r):
    """
    Calculate cost function for collaborative filtering linear regression

    :param coefficients: (k x n) vector of regression coefficients, k - users, n - features
    :param x: (m x n) product features matrix, m - products, n - features
    :param y: (m x k) results matrix, m - products, k - user ratings
    :param r: (m x k) binary matrix, 1 if j-th user rated i-th product, 0 if not
    :return: summary cost
    """
    diff = (x @ coefficients.transpose() - y)  # m x k
    return np.sum(1 / 2 * np.multiply(np.power(diff, 2), r))


def regularized_cost_function(cx, y, regularization_rate, matrices_shapes, r):
    """
    Regularized collaborative filtering linear regression cost function

    :param cx: (k*n + m*n) unrolled vector of coefficients and features
    :param y: (m x k) results matrix, m - products, k - user ratings
    :param regularization_rate: regularization rate
    :param matrices_shapes: coefficients and x original shapes
    :param r: (m x k) binary matrix, 1 if j-th user rated i-th product, 0 if not
    :return: summary cost
    """
    coefficients, x = roll_vector_to_list_of_matrices(cx, matrices_shapes)

    return cost_function(coefficients, x, y, r) \
           + regularization_rate / 2 * np.sum(np.power(coefficients, 2))\
           + regularization_rate / 2 * np.sum(np.power(x, 2))


def cost_function_derivative_c(coefficients, x, y, r):
    """
    Calculate collaborative filtering linear regression cost function
    derivative dJ/dC for each coefficient

    :param coefficients: (k x n) vector of regression coefficients, k - users, n - features
    :param x: (m x n) product features matrix, m - products, n - features
    :param y: (m x k) results matrix, m - products, k - user ratings
    :param r: (m x k) binary matrix, 1 if j-th user rated i-th product, 0 if not
    :return: (k x n) vector of derivative calculation results
    """
    return np.dot(
        np.multiply(np.subtract(
            x @ coefficients.transpose(),  # m x k
            y
        ), r).transpose(),  # k x m
        x  # m x n
    )  # k x n


def cost_function_derivative_x(coefficients, x, y, r):
    """
    Calculate collaborative filtering linear regression cost function
    derivative dJ/dx for each feature vector x

    :param coefficients: (k x n) vector of regression coefficients, k - users, n - features
    :param x: (m x n) product features matrix, m - products, n - features
    :param y: (m x k) results matrix, m - products, k - user ratings
    :param r: (m x k) binary matrix, 1 if j-th user rated i-th product, 0 if not
    :return: (m x n) vector of derivative calculation results
    """
    return np.dot(
        np.multiply(np.subtract(
            x @ coefficients.transpose(),  # m x k
            y
        ), r),  # m x k
        coefficients  # k x n
    )  # m x n


def regularized_cost_function_derivative(cx, y, regularization_rate, matrices_shapes, r):
    """
    Regularized collaborative filtering linear regression cost function derivative dJ/dC + dJ/dx

    :param cx: (k*n + m*n) unrolled vector of coefficients and features
    :param y: (m x k) results matrix, m - products, k - user ratings
    :param regularization_rate: regularization rate
    :param matrices_shapes: coefficients and x original shapes
    :return: k*n vector of derivatives
    """
    coefficients, x = roll_vector_to_list_of_matrices(cx, matrices_shapes)

    dc = cost_function_derivative_c(coefficients, x, y, r) + regularization_rate * coefficients  # k x n
    dx = cost_function_derivative_x(coefficients, x, y, r) + regularization_rate * x  # m x n

    return unroll_list_of_matrices_to_vector([dc, dx])[1]


def mean_normalize_variables(y):
    """
    Normalize variables matrix by subtracting mean from each value

    :param y: (m x k) matrix
    :return: (m x 1) vector of means, (m x k) normalized matrix
    """
    means = np.mean(y, axis=1).reshape((y.shape[0], 1))  # m x 1

    return means, y - means
