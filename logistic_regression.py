from scipy import optimize
import numpy as np
import math


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


def learn(x, y, regularization_rate):
    x = np.hstack((np.ones(x.shape[0]).reshape(x.shape[0], 1), x))

    return optimize.fmin_cg(regularized_cost_function,
                            np.zeros(x.shape[1]),
                            fprime=regularized_cost_function_derivative,
                            args=(x, y, regularization_rate),
                            maxiter=50,
                            disp=False)


def predict(coefficients, x):
    predictions = [sigmoid(np.transpose(c) @ np.concatenate([[1], x])) >= 0.5
                   for c in coefficients]
    return predictions.index(True) if True in predictions else -1
