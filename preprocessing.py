import ast
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tqdm
import wfdb
from sklearn.model_selection import train_test_split


def load_data(path: Path, frequency: str) -> Tuple:
    """Load data from dataset folder

    ### Parameters
    1. path : Path | str
        Path to dataset folder
    2. frequency : str
        Discretisation frequency 'lr' for 100hz, 'hr' for 500hz

    ### Returns
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses
    2. labels np.ndarray (n_series, )
        Labels, 0 for 'NORM', 1 for 'MI', 2 for other
    3. sig_names List[str] (n_channels, )
        Signal (channel) names
    """
    # time series loading
    path = Path(path)
    database = pd.read_csv(path.joinpath('ptbxl_database.csv'))
    data = []
    for filename in tqdm.tqdm(database[f'filename_{frequency}']):
        data.append(wfdb.rdsamp(path.joinpath(filename)))

    # labels loading
    labels = database['scp_codes'].apply(ast.literal_eval)
    agg_df = pd.read_csv(path.joinpath('scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df['diagnostic'] == 1]

    # labels aggregation
    def aggregate_diagnostic(y_dic):
        """Aggregate diagnosis into main groups"""
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    series_labels = labels.apply(aggregate_diagnostic)
    labels = []
    for diagnosis in series_labels:
        if diagnosis == ['NORM']:
            labels.append(0)
        elif 'MI' in diagnosis:
            labels.append(1)
        else:
            labels.append(2)
    
    labels = np.array(labels)
    sig_names = data[0][1]['sig_name']
    data = np.array([data[i][0].T for i in range(len(data))])
    return data, labels, sig_names


def filter_outliers(data, labels, max_std, max_mean_range):
    """Remove outliers

    ### Parameters
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses
    2. labels np.ndarray (n_series, )
        Labels, 0 for 'NORM', 1 for 'MI'
    3. max_std float
        Max std for any channel in series
    4. max_mean_range float
        Max mean value for any channel in series

    ### Returns
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses with removed outliers
    2. labels np.ndarray (n_series, )
        Labels, 0 for 'NORM', 1 for 'MI'
    """
    acceptable = list(range(len(data)))
    for i in tqdm.trange(len(data)):
        series = data[i]
        for lead in series:
            if not (np.std(lead) < max_std and \
                -max_mean_range < np.mean(lead) < max_mean_range):
                acceptable[i] = -1
                break
    acceptable = np.array(acceptable)
    filtred_labels = labels[acceptable != -1]
    filtred_data = data[acceptable != -1]
    return filtred_data, filtred_labels


def filter_others(data, labels):
    """Remove other classes

    ### Parameters
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses
    2. labels np.ndarray (n_series, )
        Labels, 0 for 'NORM', 1 for 'MI'
    
    ### Returns
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses with removed other classes
    2. labels np.ndarray (n_series, )
        Labels, 0 for 'NORM', 1 for 'MI' with removed other classes
    """
    return data[labels != 2], labels[labels != 2]


def apply_moving_average(data, window_size):
    """Apply moving average filtration

    ### Parameters
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses
    2. window_size: int
        Kernel size for average filter
    
    ### Returns
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses after moving average
    """
    def moving_average(x, window_size):
        arr = pd.Series(x).rolling(window_size).mean().to_numpy()
        arr[0] = arr[1]
        return arr
    
    data_copy = np.copy(data)
    for i in tqdm.trange(len(data)):
        series = data[i]
        for j, lead in enumerate(series):
            data_copy[i][j] = moving_average(lead, window_size)
    return data_copy


def balance_data(data, labels):
    """Balancing data labels

    ### Parameters
    1. data np.ndarray (n_series, n_channels, n_samples)
        Time serieses
    2. labels np.ndarray (n_series, )
        Labels, 0 for 'NORM', 1 for 'MI'
    
    ### Returns
    1. data np.ndarray (n_series, n_channels, n_samples)
        Balanced time serieses 
    2. labels np.ndarray (n_series, )
        Labels, 0 for 'NORM', 1 for 'MI' with balanced classes
    """
    _, counts = np.unique(labels, return_counts=True)
    indexes_for_zeros = np.array(range(len(labels)))[labels == 0]
    indexes_for_ones = np.array(range(len(labels)))[labels == 1]
    indexes_for_zeros_sample = np.random.choice(indexes_for_zeros, counts[1], replace=False)
    balanced_labels = np.concatenate([labels[indexes_for_ones], labels[indexes_for_zeros_sample]], axis=0)
    balanced_data = np.concatenate([data[indexes_for_ones], data[indexes_for_zeros_sample]], axis=0)
    return balanced_data, balanced_labels


def preprocess_data(
        path: Path,
        frequency: str = 'lr',
        std_threshold: float = 0.65,
        mean_threshold: float = 0.05,
        ma_window_size: int = 2,
        test_size: int = 0.3,
        random_state: int = None
    ):
    print("Loading data")
    data, labels, sig_names = load_data(path, frequency)
    print("Dropping other")
    data, labels = filter_others(data, labels)
    print("Filtering outliers")
    data, labels = filter_outliers(data, labels, std_threshold, mean_threshold)
    print("Applying moving average")
    data = apply_moving_average(data, ma_window_size)
    print("Balancing data")
    data, labels = balance_data(data, labels)
    print("Train-test split")
    return train_test_split(
        data, 
        labels, 
        test_size=test_size, 
        random_state=random_state
    )
