import random
from images import *


def print_predictions_accuracy(predictions, y):
    actual_digits = y.transpose()[0]
    actual_digits[actual_digits == 10] = 0

    accuracy = sum([1 for i, j in zip(predictions, actual_digits) if i == j]) / actual_digits.size * 100

    print("Prediction accuracy = " + str(accuracy) + "%")


def predict_random_images(predictions, x):
    """
    Show random images from data set and predict values
    """
    random_indexes = [random.randint(0, x.shape[0]) for i in range(20)]

    print([predictions[i] for i in random_indexes])
    plot_image(np.hstack([vector_to_square_image(x[i]) for i in random_indexes]))
