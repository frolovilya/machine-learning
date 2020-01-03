import scipy.io
import matplotlib.pyplot as plt
import random
import math
import numpy as np


def load_data():
    data = scipy.io.loadmat('ex3data1.mat')
    x = data['X']  # m x n^2, where m - experiments count, n - square image size
    y = data['y']  # m x 1 vector of image classes (numbers 0 - 9)

    return x, y


def vector_to_square_image(image_vector):
    size = int(math.sqrt(image_vector.size))
    return np.transpose(image_vector.reshape(size, size))


def plot_image(image):
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.show()


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
