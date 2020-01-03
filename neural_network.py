import numpy as np
from logistic_regression import sigmoid


def predict_digit(layer_coefficients, input_data):
    """
    Predict neural network output for input data with given trained layer coefficients.

    :param layer_coefficients: array of layer coefficients
    :param input_data: input layer data
    :return: index of the max output
    """
    input_data = input_data.reshape(input_data.size, 1)  # n x 1

    for theta in layer_coefficients:
        input_data = sigmoid(np.dot(
            theta,  # l x (n + 1)
            np.vstack([[1], input_data])  # (n + 1) x 1
        ))  # l x 1

    max_index = np.argmax(input_data) + 1

    # data set contains 10 instead of 0
    return max_index if max_index < 10 else 0
