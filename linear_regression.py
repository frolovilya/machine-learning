import numpy as np


def cost_function(coefficients, x, y):
    """
    Calculate cost function for linear regression

    :param coefficients: (1 x n) vector of regression coefficients
    :param x: (m x n) matrix, m - experiments count, n - variables count
    :param y: (m x 1) results vector
    :return: summary cost
    """
    coefficients = coefficients.reshape(coefficients.size, 1)
    diff = (x @ coefficients - y)
    return 1 / (2 * y.shape[0]) * (diff.transpose() @ diff)[0, 0]


def regularized_cost_function(coefficients, x, y, regularization_rate):
    c = np.reshape(coefficients[1:coefficients.size], (coefficients.size - 1, 1))

    return cost_function(coefficients, x, y) \
           + regularization_rate / (2 * y.shape[0]) * (c.transpose() @ c)[0, 0]


def cost_function_derivative(coefficients, x, y):
    """
    Calculate linear regression cost function derivative dJ/dC for each coefficient

    :param coefficients: (1 x n) vector of regression coefficients
    :param x: input variables (m x n) matrix,
              n - variables count, m - experiments count
    :param y: output results (m x 1) vector, m - experiments count
    :return: (1 x n) vector of derivative calculation results
    """
    coefficients = coefficients.reshape(coefficients.size, 1)  # n x 1

    return 1 / y.shape[0] * np.dot(
        np.subtract(
            x @ coefficients,
            y
        ).transpose(),  # 1 x m
        x  # m x n
    )[0]  # 1 x n


def regularized_cost_function_derivative(coefficients, x, y, regularization_rate):
    c = np.concatenate([[0], coefficients[1:coefficients.size]])

    return cost_function_derivative(coefficients, x, y) \
           + regularization_rate / y.shape[0] * c


def normal_equation(x, y):
    """
    Calculate linear regression coefficients analytically using normal equation

    :param x: input variables (m x (n - 1)) matrix,
              n - variables count, m - experiments count
    :param y: output results (m x 1) vector, m - experiments count
    :return: 1 x n vector of linear coefficients
    """
    coefficients = np.linalg.pinv(x.transpose() @ x) @ x.transpose() @ y
    return coefficients.transpose()[0]
