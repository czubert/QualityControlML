import os
import pandas as pd
from joblib import dump, load
import numpy as np
import scipy.linalg as LA
import math


def baseline(y, deg=None, max_it=None, tol=None):
    """
    Computes the baseline of a given data.

    Iteratively performs a polynomial fitting in the data to detect its
    baseline. At every iteration, the fitting weights on the regions with
    peaks are reduced to identify the baseline only.

    Parameters
    ----------
    y : ndarray
        Data to detect the baseline.
    deg : int (default: 3)
        Degree of the polynomial that will estimate the data baseline. A low
        degree may fail to detect all the baseline present, while a high
        degree may make the data too oscillatory, especially at the edges.
    max_it : int (default: 100)
        Maximum number of iterations to perform.
    tol : float (default: 1e-3)
        Tolerance to use when comparing the difference between the current
        fit coefficients and the ones from the last iteration. The iteration
        procedure will stop when the difference between them is lower than
        *tol*.

    Returns
    -------
    ndarray
        Array with the baseline amplitude for every original point in *y*
    """
    # for not repeating ourselves in `envelope`
    if deg is None: deg = 3
    if max_it is None: max_it = 100
    if tol is None: tol = 1e-3

    order = deg + 1
    coeffs = np.ones(order)

    # try to avoid numerical issues
    cond = math.pow(abs(y).max(), 1. / order)
    x = np.linspace(0., cond, y.size)
    base = y.copy()

    vander = np.vander(x, order)
    vander_pinv = LA.pinv(vander)

    for _ in range(max_it):
        coeffs_new = np.dot(vander_pinv, y)

        if LA.norm(coeffs_new - coeffs) / LA.norm(coeffs) < tol:
            break

        coeffs = coeffs_new
        base = np.dot(vander, coeffs)
        y = np.minimum(y, base)

    return coeffs

# constants
PEAKS = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['471', '581'],
    'peak2': ['1154', '1221'],
    'peak3': ['1535', '1688'],
    'peak4': ['1154', '1215'],
    'peak5': ['1453', '1510'],
}


def save_as_joblib(data_to_save, file_name, path):
    if not os.path.isdir(f'{path}'):
        os.makedirs(f'{path}')

    dump(data_to_save, f'{path}/{file_name}.joblib')


def read_joblib(file_name, dir_name):
    return load(f'{dir_name}/{file_name}.joblib')


def save_as_csv(data, file_name, dir_name):
    if not os.path.isdir(f'{dir_name}'):
        os.mkdir(f'{dir_name}')

    data.to_csv(f'{dir_name}/{file_name}.csv')


def read_csv(file_name, dir_name):
    return pd.read_csv(f'{dir_name}/{file_name}.csv')


def change_col_names_type_to_str(df):
    df.copy()
    cols = [str(x) for x in df.columns]
    df.columns = cols
    return df
