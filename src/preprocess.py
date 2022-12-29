"""
Preprocessing functions
"""

import numpy as np
import pandas as pd


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


def represent(time_series: np.array) -> pd.DataFrame:
    """

    :param time_series:
    :return:
    """
    res = []
    features = time_series.shape[2]
    print(features)
    for i in range(time_series.shape[0]):
        tmp_res = []
        for feature in range(features):
            to_series = pd.Series(time_series[i].T[feature].copy())
            tmp_res.append(to_series)
        res.append(tmp_res)
    return pd.DataFrame(res, columns=[f'feature {i}' for i in range(1, 13)])
