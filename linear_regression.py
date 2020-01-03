import numpy as np


def cost_function(coefficients, x, y):
    """
    Calculate cost function for linear regression

    @param coefficients: (1 x n) vector of regression coefficients
    @param x: (m x n) matrix, m - experiments count, n - variables count
    @param y: (m x 1) results vector
    @return: summary cost
    """
    coefficients = coefficients.reshape(coefficients.size, 1)
    diff = (x @ coefficients - y)
    return 1 / (2 * y.size) * (diff.transpose() @ diff)[0, 0]


def cost_function_derivative(coefficients, x, y):
    """
    @param x: input variables (m x (n - 1)) matrix,
              n - variables count, m - experiments count
    @param y: output results (m x 1) vector, m - experiments count
    @return: (n x 1) vector of derivative calculation results,
             n - variables (coefficients) count
    """
    coefficients = coefficients.reshape(coefficients.size, 1)

    return np.array([1 / y.size * np.dot(
        np.subtract(
            x @ coefficients,
            y
        ).transpose(),  # 1 x m
        x[:, index:(index + 1)]  # m x 1
    )[0, 0] for index in range(coefficients.size)])


def normal_equation(x, y):
    """
    Calculate linear regression coefficients analytically using normal equation

    @param x: input variables (m x (n - 1)) DataFrame,
              n - variables count, m - experiments count
    @param y: output results (m x 1) DataFrame, m - experiments count
    @return: n x 1 vector of linear coefficients
    """
    coefficients = np.linalg.pinv(x.transpose() @ x) @ x.transpose() @ y
    return coefficients.transpose()[0]
