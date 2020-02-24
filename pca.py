import numpy as np


def normalize_variable(x):
    """
    :param x: (m x n) vector,
               m - experiments count, n - variables
    :return: normalized (m x n) vector, mu - column means, sigma - column std deviations
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x - mu, axis=0)

    return [np.divide(x - mu, sigma), mu, sigma]


def cov_matrix(x):
    return 1 / x.shape[0] * (x.transpose() @ x)


def find_eig_vectors(x):
    return np.linalg.eig(cov_matrix(x))


def project_data(x, eigenvectors, new_dimension):
    """
    Reduce X dimensionality from n to k

    :param x: (m x n) data set
    :param eigenvectors: (n x n) principal components of x
    :param new_dimension: new data dimension k < n
    :return: (m x k) data
    """
    eigenvectors = eigenvectors[:, 0:new_dimension]  # n x k

    return x @ eigenvectors  # m x k


def recover_data(x, eigenvectors):
    """
    Recover data from k dimension to n

    :param x: (m x k) data with reduced dimension
    :param eigenvectors: original data eigenvectors (n x n)
    :return: (m x n) recovered data
    """
    return x @ eigenvectors[:, 0:x.shape[1]].transpose()  # (m x k) @ (k x n)
