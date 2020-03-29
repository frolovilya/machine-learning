import re
import numpy as np


def process_email(file_path):
    """
    Process email body to classify with SVM

    :param file_path: email file path
    :return: processed body text
    """
    with open(file_path, 'r') as file:
        email_str = file.read().lower()  # to lower case
        email_str = " ".join(email_str.split())  # remove whitespaces
        email_str = re.sub('[$]+', 'dollar ', email_str)  # replace $ sign
        email_str = re.sub('(http|https)://[^\s]*', 'httpaddr', email_str)  # replace http links
        email_str = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_str)  # replace email addresses
        email_str = re.sub('[^0-9a-zA-Z]+', ' ', email_str)  # remove non-words
        email_str = re.sub('[0-9]+', 'number', email_str)  # replace numbers
        return email_str.split()


def get_vocab_vector(vocab_file):
    """
    Load vocab file as vector

    :param vocab_file: spam vocabulary dictionary file
    :return: vocab 1 x n vector, n - number of words
    """
    return np.loadtxt(vocab_file, dtype=str)[:, 1]


def map_text_to_feature_vector(email_body, vocab):
    """
    Map email body text vector to vocab vector

    :param email_body: 1 x m vector, m - number of words in email
    :param vocab: 1 x n vector, n - number of vocab words
    :return: 1 x n boolean vector of matched words
    """
    return np.array([v in email_body for v in vocab], dtype=int, ndmin=2)

