import numpy as np
import pandas as pd


def normalize_variable(Xj):
    """
    @param Xj: (m x 1) vector,
               m - experiments count, 1 - one variable
    @return: normalized (m x 1) vector
    """
    return (Xj.values - np.mean(Xj.values)) / np.std(Xj.values)


def concat_with_x0(x):
    """
    @param x: (m x (n - 1)) DataFrame,
              n - variables count, m - experiments count
    @return: (m x n) DataFrame, with first column filled with 1
    """
    return pd.concat([pd.DataFrame([1] * x.shape[0]), x], axis=1)

