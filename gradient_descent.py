import numpy as np


def gradient_descent(coefficients_count: int,
                     hypothesis_function_derivative,
                     args,
                     learning_rate: float,
                     max_iterations: int = 1000):
    """
    Calculate regression coefficients in vector form

    :param coefficients_count: regression coefficients count
    :param hypothesis_function_derivative: first derivative dJ/dCj of hypothesis function J(C)
    :param args: hypothesis function derivative arguments
    :param learning_rate: gradient descent alpha parameter
    :param max_iterations: max algorithm iterations until convergence
    :return: 1 x n vector of linear coefficients
    """
    # 1 x n vector
    coefficients = np.zeros((1, coefficients_count))

    for i in range(max_iterations):
        coefficients = coefficients - learning_rate \
                       * hypothesis_function_derivative(coefficients, *args)

    return coefficients[0]
