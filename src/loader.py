"""
Load dataset
"""

import pandas as pd
import numpy as np
import wfdb
import ast

PATH = 'plt/'


def _load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def _aggregate_diagnostic(y_dic):
    agg_df = pd.read_csv(PATH + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def _add_is_mi(y: pd.DataFrame) -> pd.DataFrame:
    res = []
    for item in y['diagnostic_superclass']:
        if item == ['NORM']:
            res.append(0)
            continue
        if item == ['MI']:
            res.append(1)
            continue
        res.append(np.nan)
    y['is_MI'] = res

    # Delete what's not NORM or not MI
    y = y.drop(y[y.is_MI.isna()].index)
    return y


def get_data() -> tuple:
    """
    :return: Tuple of (X, Y)
    """
    sampling_rate = 100

    # load and convert annotation data
    Y = pd.read_csv(PATH + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = _load_raw_data(Y, sampling_rate, PATH)

    # Load scp_statements.csv for diagnostic aggregation

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(_aggregate_diagnostic)
    Y = _add_is_mi(Y)
    return X, Y
