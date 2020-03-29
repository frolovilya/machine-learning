from scipy import optimize
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


def find_coefficients(x, y,
                      regularized_cost_function, regularized_cost_function_derivative,
                      regularization_rate=0,
                      max_iterations=100,
                      initial_coefficients=None,
                      additional_args=()):
    """
    Find linear regression coefficients by minimizing regularized_cost_function

    :param x: (m x n) matrix of feature parameters, including x0
    :param y: (m x 1) vector of actual results
    :param regularized_cost_function: cost function
    :param regularized_cost_function_derivative: cost function derivative
    :param regularization_rate: regularization rate
    :param max_iterations: max algorithm iterations
    :param initial_coefficients: initial coefficient values, or zeros vector if None
    :param additional_args: additional arguments
    :return: linear regression coefficients (n x 1) vector
    """
    np.seterr(divide='ignore', invalid='ignore')
    return optimize.fmin_cg(regularized_cost_function,
                            np.zeros(x.shape[1]) if initial_coefficients is None else initial_coefficients,
                            fprime=regularized_cost_function_derivative,
                            args=(x, y, regularization_rate) + additional_args,
                            maxiter=max_iterations,
                            disp=False)
