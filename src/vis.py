"""
Visualization functions
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_series(series: pd.DataFrame, size=(25, 70), rows=12):
    """
    Draw ecg series
    :return: Figure
    """
    if series.shape[1] != 12:
        raise ValueError('Frame columns must be 12')

    if rows > 12:
        raise ValueError('Rows must be less or equal than 12')

    fig, axs = plt.subplots(rows, figsize=size)
    all_axs = [axs[i] for i in range(rows)]
    curr_ax = 0
    for ser in series.T:
        sns.lineplot(ser, ax=all_axs[curr_ax])
        curr_ax += 1

    return fig
