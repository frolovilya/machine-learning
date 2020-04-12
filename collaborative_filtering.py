import numpy as np
from utils import roll_vector_to_list_of_matrices, unroll_list_of_matrices_to_vector
from optimize import find_coefficients


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


def regularized_cost_function(cx, x, y, regularization_rate, matrices_shapes, r):
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
           + regularization_rate / 2 * np.sum(np.power(coefficients, 2)) \
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


def regularized_cost_function_derivative(cx, x, y, regularization_rate, matrices_shapes, r):
    """
    Regularized collaborative filtering linear regression cost function derivative dJ/dC + dJ/dx

    :param cx: (k*n + m*n) unrolled vector of coefficients and features
    :param y: (m x k) results matrix, m - products, k - user ratings
    :param regularization_rate: regularization rate
    :param matrices_shapes: coefficients and x original shapes
    :param r: (m x k) binary matrix, 1 if j-th user rated i-th product, 0 if not
    :return: k*n vector of derivatives
    """
    coefficients, x = roll_vector_to_list_of_matrices(cx, matrices_shapes)

    dc = cost_function_derivative_c(coefficients, x, y, r) + regularization_rate * coefficients  # k x n
    dx = cost_function_derivative_x(coefficients, x, y, r) + regularization_rate * x  # m x n

    return unroll_list_of_matrices_to_vector([dc, dx])[1]


def mean_normalize_variables(y, r):
    """
    Normalize variables matrix by subtracting mean from each value

    :param y: (m x k) matrix
    :param r: (m x k) binary matrix, 1 if j-th user rated i-th product, 0 if not
    :return: (m x 1) vector of means, (m x k) normalized matrix
    """
    means = (np.sum(y, axis=1) / np.sum(r, axis=1)).reshape((y.shape[0], 1))  # m x 1

    return means, np.multiply(y - means, r)


def find_new_parameters(y_norm, r):
    """
    Use collaborative filtering algorithm to find client's theta and movie's X feature vectors

    :param y_norm: (m x k) mean-normalized movie ratings
    :param r: binary (m x k) matrix indicating if user rated movie
    :return: (k x n) coefficients, k - users, n - features;
             (m x n) movie features, m - movies, n - features
    """
    num_features = 10
    theta = np.random.rand(y_norm.shape[1], num_features)  # k x n
    x = np.random.rand(y_norm.shape[0], num_features)  # m x n

    original_shapes, cx = unroll_list_of_matrices_to_vector([theta, x])

    optimal_values = find_coefficients((), y_norm,
                                       regularized_cost_function, regularized_cost_function_derivative,
                                       regularization_rate=10,
                                       max_iterations=200,
                                       initial_coefficients=cx,
                                       additional_args=(original_shapes, r))

    new_theta, new_x = roll_vector_to_list_of_matrices(optimal_values, original_shapes)

    return new_theta, new_x


def find_recommended_movies(x, coefficients, y_means):
    """
    Find recommended movies using collaborative filtering calculation results

    :param x: (m x n) movie features, m - movies, n - features
    :param coefficients: (n x 1) user coefficients, n - features
    :param y_means: (m x 1) mean movie ratings, m - movies
    :return: recommended items sorted by descending rating
    """
    my_recommendations = x @ coefficients + y_means  # m x 1

    # add indexes column
    my_recommendations = np.hstack((
        np.arange(my_recommendations.shape[0], dtype=int).reshape(my_recommendations.shape),
        my_recommendations
    ))  # m x 2

    # desc sort by rating
    return my_recommendations[np.argsort(my_recommendations[:, 1])][::-1]

