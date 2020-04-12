import random
import numpy as np
from images import vector_to_square_image, plot_image


def print_predictions_accuracy(predictions, y):
    """
    Calculate image class prediction accuracy

    :param predictions: 1 x m vector of predictions for each image
    :param y: m x 1 vector of actual image classes
    """
    actual_digits = y.transpose()[0]
    actual_digits[actual_digits == 10] = 0

    accuracy = sum([1 for i, j in zip(predictions, actual_digits) if i == j]) / actual_digits.size * 100

    print("Prediction accuracy = " + str(accuracy) + "%")


def predict_random_images(predictions, x):
    """
    Show random images from data set and their predicted classes

    :param predictions: 1 x m vector of predictions for each image
    :param x: m x n^2 matrix, where m - experiments count, n - square image size
    """
    random_indexes = [random.randint(0, x.shape[0]) for i in range(20)]

    print([predictions[i] for i in random_indexes])
    plot_image(np.hstack([vector_to_square_image(x[i]) for i in random_indexes]))
