import numpy as np
import scipy.io
from scipy import optimize
from logistic_regression import sigmoid
from image_recognition import *


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


def roll_vector_to_list_of_matrices(v, shapes):
    first = 0
    result = []

    for shape in shapes:
        last = first + shape[0] * shape[1]
        result.append(np.array(v[first:last]).reshape(shape[0], shape[1]))
        first = last

    return result


def unroll_list_of_matrices_to_vector(l):
    result_vector = np.array([])
    original_shape = []

    for m in l:
        original_shape.append(m.shape)
        result_vector = np.hstack([
            result_vector,
            m.reshape(m.shape[0] * m.shape[1])
        ])

    return original_shape, result_vector


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


def nn_regularized_cost_function(unrolled_layer_coefficients, shape, x, y, regularization_rate):
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


def nn_regularized_gradient(unrolled_layer_coefficients, shape, x, y, regularization_rate):
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


if __name__ == "__main__":
    data = scipy.io.loadmat('data/ex3data1.mat')
    x = data['X']  # m x n^2, where m - experiments count, n - square image size
    y = data['y']  # m x 1 vector of image classes (numbers 0 - 9)

    weights = scipy.io.loadmat('data/ex3weights.mat')
    nn_coefficients = (
        weights['Theta1'],  # S1 x (n^2 + 1), where S1 - hidden layer size, n - square image size
        weights['Theta2']  # SL x (S1 + 1), where SL - output layer size, S1 - hidden layer size
    )


    def digit_to_output_vector(digit):
        """
        Returns 10 x 1 vector with all 0 except 1 for index corresponding to the provided digit.
        If digit == 0, then 10th element == 1.
        """
        out = np.zeros(10)
        out[9 if digit == 0 else digit - 1] = 1
        return out


    def predict_digit(nn_coefficients, image):
        output_data = forward_propagation(nn_coefficients, image)[-1]
        max_index = np.argmax(output_data) + 1

        # data set contains 10 instead of 0
        return max_index if max_index < 10 else 0


    expected_output = np.array([digit_to_output_vector(d) for d in y]).transpose()

    original_shape = ((25, 401), (10, 26))
    initial_coefficients = np.random.uniform(-1, 1, 25 * 401 + 10 * 26)

    # original_shape, initial_guess = unroll_list_of_matrices_to_vector(nn_coefficients)

    coefficients_vector = optimize.fmin_cg(nn_regularized_cost_function,
                                           initial_coefficients,
                                           fprime=nn_regularized_gradient,
                                           args=(original_shape, x.transpose(), expected_output, 0),
                                           maxiter=50)

    nn_learned_coefficients = roll_vector_to_list_of_matrices(coefficients_vector, original_shape)

    print(nn_learned_coefficients)

    nn_new_predictions = [predict_digit(nn_learned_coefficients, image.reshape(image.size, 1)) for image in x]
    print_predictions_accuracy(nn_new_predictions, y)
