import numpy as np


def init_cluster_centroids(x, number_of_clusters):
    """
    Randomly assign cluster centroids to X values

    :param x: (m x n) matrix, m - experiments count, n - features count
    :param number_of_clusters: number of clusters to assign
    :return: (number_of_clusters x n) vector of cluster centroids
    """
    return x[np.random.choice(x.shape[0], number_of_clusters, replace=False), :]


def find_closest_centroid(x, centroids):
    """
    Find closest centroid to every value in X

    :param x: (m x n) matrix, m - experiments count, n - features count
    :param centroids: (k x n) matrix of k cluster centroids
    :return: (1 x m) vector of assigned centroid indexes 0..(k - 1)
    """

    return np.argmin([np.linalg.norm(x - centroids[k], axis=1)
                      for k in range(centroids.shape[0])], axis=0)


def move_centroids(centroids, x, assigned_clusters):
    """
    Move centroids to their cluster means

    :param centroids: (k x n) matrix, k - centroids, n - features
    :param x: (m x n) matrix of X vectors
    :param assigned_clusters: (1 x m) vector of assigned centroid indexes 0..(k - 1)
    :return (k x n) new centroid values
    """

    return np.array([1 / np.sum(assigned_clusters == k) * np.sum(x[assigned_clusters == k], axis=0)
                     for k in range(centroids.shape[0])])


def find_clusters(x, number_of_clusters):
    """
    Find number_of_clusters clusters in X

    :param x: (m x n) matrix data set of m vectors
    :param number_of_clusters: number of clusters to find
    :return: ((1 x m), (k x n)) assigned clusters vector, centroids matrix
    """
    centroids = init_cluster_centroids(x, number_of_clusters)  # k x n
    clusters = np.zeros((1, x.shape[0]))  # 1 x m
    find = True

    while find:
        clusters = find_closest_centroid(x, centroids)  # m x 1
        new_centroids = move_centroids(centroids, x, clusters)  # k x n

        # use l2 norm to calculate how far cluster centroids were moved from the previous location
        dist = np.linalg.norm(centroids - new_centroids)
        find = dist > 0

        centroids = new_centroids

    return clusters, centroids
