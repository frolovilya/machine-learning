import math
import numpy as np
import matplotlib.pyplot as plt


def vector_to_square_image(image_vector):
    size = int(math.sqrt(image_vector.size))
    return np.transpose(image_vector.reshape(size, size))


def plot_image(image):
    plt.figure(figsize=(10, 1))
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.show()
