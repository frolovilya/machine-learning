import numpy as np
import scipy.io
from scipy import optimize
from logistic_regression import sigmoid
from image_recognition import *
from utils import *


def forward_propagation(layer_coefficients, input_data):
    """
    Calculate neural network output based on input data and layer coefficients.
    Forward propagation algorithm.

    :param layer_coefficients: 1 x (L - 1) array of layer coefficients vectors, where L - layers count
    :param input_data: S0 x m input layer vector, where S0 - input layer units count, m - experiments count
    :return: 1 x l vector of layer activation vectors Sl x m, where Sl - l'th layer units count,
             m - experiments count
    """
    data = [input_data]  # S0 x m

    for theta in layer_coefficients:
        data.append(
            sigmoid(np.dot(
                theta,  # Sl x (S[l-1] + 1)
                np.vstack(([np.ones(data[-1].shape[1])], data[-1]))  # (S[l-1] + 1) x m
            ))  # Sl x m
        )

    return data


def nn_cost_function(layer_coefficients, x, y):
    """
    Calculate cost function for neural network

    :param layer_coefficients: 1 x (L - 1) array of layer coefficients vectors, where L - layers count
    :param x: S0 x m input layer vector, where S0 - input layer units count, m - experiments count
    :param y: SL x m expected results matrix, where Sl - output layer units count, m - experiments count
    :return: summary cost
    """
    return - 1 / y.shape[1] * np.sum((
            np.multiply(y, np.log(forward_propagation(layer_coefficients, x)[-1]))  # SL x m
            +
            np.multiply((1 - y), np.log(1 - forward_propagation(layer_coefficients, x)[-1]))  # SL x m
    ))


def nn_regularized_cost_function(unrolled_layer_coefficients, x, y, regularization_rate, shape):
    """
    Regularized neural network cost function.
    See nn_cost_function description.
    """
    layer_coefficients = roll_vector_to_list_of_matrices(unrolled_layer_coefficients, shape)

    cost = nn_cost_function(layer_coefficients, x, y)

    for theta in layer_coefficients:
        unrolled_theta = theta.reshape(theta.shape[0] * theta.shape[1], 1)

        cost += regularization_rate / (2 * y.shape[1]) \
                * (unrolled_theta.transpose() @ unrolled_theta)[0, 0]

    return cost


def sigmoid_derivative(z):
    """
    Calculate sigmoid function derivative dg/dz
    """
    return np.multiply(z, 1 - z)


def back_propagation(layer_coefficients, y, output):
    """
    Calculate error delta values for each layer and unit

    :param layer_coefficients: 1 x (L - 1) array of layer coefficients vectors, where L - layers count
    :param y: SL x m expected results matrix, where SL - output layer units count, m - experiments count
    :param output: 1 x l vector of layer activation vectors Sl x m, where Sl - l'th layer units count,
             m - experiments count
    :return: 1 x (L - 1) vector of Sl x m delta values
    """
    delta = [output[-1] - y]

    for l in reversed(range(1, len(layer_coefficients))):
        delta.insert(
            0,
            np.multiply(
                np.dot(
                    layer_coefficients[l].transpose(),  # (Sl + 1) x S[l + 1]
                    delta[0]  # S[l + 1] x m
                )[1:, :],  # Sl x m
                sigmoid_derivative(output[l])  # Sl x m
            )
        )

    return delta


def nn_gradient(layer_coefficients, x, y):
    """
    Neural network gradient (derivative) function

    :param layer_coefficients: 1 x (L - 1) array of layer coefficients vectors, where L - layers count
    :param x: S0 x m input layer vector, where S0 - input layer units count, m - experiments count
    :param y: SL x m expected results matrix, where SL - output layer units count, m - experiments count
    :return: 1 x (L - 1) vector of S[l + 1] x (Sl + 1) gradient values
    """
    output = forward_propagation(layer_coefficients, x)  # l, Sl x m
    deltas = back_propagation(layer_coefficients, y, output)  # l, Sl x m

    grad = []

    for l in range(len(deltas)):
        grad.append(
            1 / y.shape[1] * np.dot(
                deltas[l],  # S[l + 1] x m
                np.vstack([np.ones(output[l].shape[1]), output[l]]).transpose()  # m x (Sl + 1)
            )  # S[l + 1] x (Sl + 1)
        )

    return grad


def nn_regularized_gradient(unrolled_layer_coefficients, x, y, regularization_rate, shape):
    """
    Regularized neural network gradient.
    See nn_gradient description.
    """
    layer_coefficients = roll_vector_to_list_of_matrices(unrolled_layer_coefficients, shape)

    gradients = nn_gradient(layer_coefficients, x, y)

    reg_gradients = []

    for l in range(len(layer_coefficients) - 1):
        reg_gradients.append(
            gradients[l]  # S[l + 1] x (Sl + 1)
            + regularization_rate / y.shape[1]
            * np.hstack([
                np.zeros((layer_coefficients[l].shape[0], 1)),
                layer_coefficients[l][:, 1:]
            ])  # S[l + 1] x (Sl + 1)
        )

    reg_gradients.append(gradients[-1])

    return unroll_list_of_matrices_to_vector(reg_gradients)[1]

