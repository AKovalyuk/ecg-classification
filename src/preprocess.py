"""
Preprocessing functions
"""

import numpy as np


def train_split(x, y, test_fold: int) -> tuple:
    """
    Split on train and test
    :param x: X data
    :param y: Y data
    :param test_fold: Each n value will be for testing
    :return: (X_train, y_train, X_test, y_test)
    """

    X_train = x[np.where(y.strat_fold != test_fold)]
    y_train = y[(y.strat_fold != test_fold)].is_MI

    # Test
    X_test = x[np.where(y.strat_fold == test_fold)]
    y_test = y[y.strat_fold == test_fold].is_MI

    return X_train, y_train, X_test, y_test
