import math
import numpy as np
import matplotlib.pyplot as plt


def vector_to_square_image(image_vector):
    """
    Reshape image n^2 vector to n x n image matrix

    :param image_vector: 1 x n^2 image data vector
    :return: n x n image matrix
    """
    size = int(math.sqrt(image_vector.size))
    return np.transpose(image_vector.reshape(size, size))


def plot_image(image):
    """
    Plot image data

    :param image: a x b image data
    """
    plt.figure(figsize=(10, 1))
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.show()
