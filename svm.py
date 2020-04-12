from sklearn import svm


def fit_svm(x, y, kernel, C=1, gamma=1, probability=False):
    """
    Fit SVM classifier with given parameters

    :param x: (m x n) data, m - experiments count, n - features
    :param y: (m x 1) vector of experiment results
    :param kernel: kernel type, ex "linear" or "rbf" (gaussian)
    :param C: regularization rate (1/lambda)
    :param gamma: gaussian kernel coefficient
    :param probability: include probability information (used for clf.predict_proba method)
    :return: SVM classifier
    """
    clf = svm.SVC(C=C, gamma=gamma, max_iter=max(C, 1) * 10000,
                  kernel=kernel, probability=probability)
    clf.fit(x, y.ravel())
    return clf
