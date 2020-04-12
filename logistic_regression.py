import numpy as np
import math
from utils import concat_with_x0
from optimize import find_coefficients


def sigmoid(z):
    """
    Calculate sigmoid function

    @param z: vector, e^z
    @return: vector of sigmoid function results
    """
    return 1 / (1 + np.power(math.e, -z))


def cost_function(coefficients, x, y):
    """
    Calculate cost function for logistic regression

    @param coefficients: (1 x n) vector of regression coefficients
    @param x: (m x n) matrix, m - experiments count, n - variables count
    @param y: (m x 1) results vector
    @return: summary cost
    """
    coefficients = coefficients.reshape(coefficients.size, 1)

    return - 1 / y.size * (
            y.transpose() @ np.log(sigmoid(x @ coefficients))
            +
            (1 - y).transpose() @ np.log(1 - sigmoid(x @ coefficients))
    )[0, 0]


def regularized_cost_function(coefficients, x, y,
                              regularization_rate: float):
    c = np.reshape(coefficients[1:coefficients.size], (coefficients.size - 1, 1))

    return cost_function(coefficients, x, y) \
           + regularization_rate / (2 * y.size) * (c.transpose() @ c)[0, 0]


def cost_function_derivative(coefficients, x, y):
    """
    Calculate cost function first derivative dJ/dC for each coefficient

    @param coefficients: (1 x n) coefficients vector
    @param x: input variables (m x (n - 1)) matrix,
              n - variables count, m - experiments count
    @param y: results (m x 1) vector, m - experiments count
    @return: (n x 1) vector of derivative calculation results for each coefficient
    """
    coefficients = coefficients.reshape(coefficients.size, 1)

    return 1 / y.size * np.dot(
        np.subtract(
            sigmoid(x @ coefficients),
            y
        ).transpose(),  # 1 x m
        x  # m x n
    )[0]


def regularized_cost_function_derivative(coefficients, x, y,
                                         regularization_rate: float):
    c = np.concatenate([[0], coefficients[1:coefficients.size]])

    return cost_function_derivative(coefficients, x, y) \
           + regularization_rate / y.size * c


def fit_logistic_regression(x, y, regularization_rate, max_iter=100):
    x = concat_with_x0(x)

    return find_coefficients(x, y,
                             regularized_cost_function, regularized_cost_function_derivative,
                             regularization_rate, max_iter)


def predict_digit(coefficients, x):
    """
    Find logistic regression classifier that predicts true for a given x

    :param coefficients: m x n matrix, m - classifiers count, n - coefficients count
    :param x: n x 1 feature vector
    :return: classifier index [0..m] or -1
    """
    predictions = [sigmoid(np.transpose(c) @ np.concatenate([[1], x])) >= 0.5
                   for c in coefficients]
    return predictions.index(True) if True in predictions else -1


def map_features(x1, x2, degree):
    """
    Map features to polynomial

    @param x1: (m x 1) vector of variable x1 values
    @param x2: (m x 1) vector of variable x2 values
    @param degree: polynomial degree
    @return: (m x N) matrix with N polynomial features
    """
    return np.concatenate([[np.power(x1, i - j) * np.power(x2, j)
                            for j in range(i + 1)]
                           for i in range(1, degree + 1)]).transpose()[0]
