import numpy as np
import math


def get_mean(data):
    """
    Calculate data feature means

    :param data: (m x n) matrix, m - examples, n - features
    :return: (1 x n) means vector
    """
    return np.mean(data, axis=0)


def get_variance(data):
    """
    Calculate data feature variances

    :param data: (m x n) matrix, m - examples, n - features
    :return: (1 x n) variances vector
    """
    return np.var(data, axis=0)


def get_covariance(data):
    """
    Calculate data covariance matrix

    :param data: (m x n) matrix, m - examples, n - features
    :return: (n x n) covariance matrix
    """
    mean = get_mean(data)  # 1 x n

    return 1 / data.shape[0] * (data - mean).transpose() @ (data - mean)


def get_probability(data, mean, variance):
    """
    Calculate probability for each sample in dataset by fitting it to the normal distribution
    defined by dataset's mean and variance

    :param data: (m x n) matrix, m - examples, n - features
    :param mean: (1 x n) mean of each feature
    :param variance: (1 x n) variance of each feature
    :return: (m x 1) probability vector for each sample in dataset
    """
    probability = 1 / np.sqrt(2 * math.pi * variance) \
                  * np.exp(- np.power(data - mean, 2) / (2 * variance))  # m x n

    return np.product(probability, axis=1)  # m x 1


def get_multivariate_probability(data, mean, covariance_matrix):
    """
    Calculate multivariate probability for each sample in dataset by fitting it to the normal distribution
    defined by dataset's mean and features covariance matrix

    :param data: (m x n) matrix, m - examples, n - features
    :param mean: (1 x n) mean of each feature
    :param covariance_matrix: (n x n) covariance matrix
    :return: (m x 1) probability vector for each sample in dataset
    """
    p = [1 / (math.pow((2 * math.pi), data[i, :].size / 2) * np.power(np.linalg.det(covariance_matrix), 1/2)) \
         * np.exp(- 1/2 * (data[i, :] - mean) @ np.linalg.inv(covariance_matrix) @ (data[i, :] - mean).transpose())
         for i in range(data.shape[0])]

    return np.array(p)


def f1_score(probabilities, threshold, y):
    """
    Calculate F1 score for classifying probabilities as anomaly with given threshold

    :param probabilities: (m x 1) vector of sample probabilities
    :param threshold: upper probability bound to classify a sample as anomaly
    :param y: (m x 1) logic vector of true labels
    :return: F1 score value
    """
    classified_samples = probabilities < threshold

    true_positive = np.intersect1d(np.argwhere(classified_samples == True), np.argwhere(y == 1)).size
    false_positive = np.intersect1d(np.argwhere(classified_samples == True), np.argwhere(y == 0)).size
    false_negative = np.intersect1d(np.argwhere(classified_samples == False), np.argwhere(y == 1)).size

    zero_division = math.pow(10, -7)
    precision = true_positive / (true_positive + false_positive + zero_division)
    recall = true_positive / (true_positive + false_negative + zero_division)

    return 2 * precision * recall / (precision + recall + zero_division)


def select_threshold(probabilities, y):
    """
    Find optimal threshold to classify data as anomaly

    :param probabilities: (m x 1) vector of sample probabilities
    :param y: (m x 1) known data labels (0 - normal, 1 - anomaly)
    :return: threshold value, f1 score
    """
    thresholds = np.linspace(min(probabilities), max(probabilities), 1000)
    results = np.array([(t, f1_score(probabilities, t, y)) for t in thresholds])

    return results[np.argmax(results[:, 1])]

