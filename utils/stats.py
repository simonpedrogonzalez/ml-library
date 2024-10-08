import numpy as np
import pandas as pd

def unique_proba(y: np.ndarray, w: np.ndarray):
    # assume int array and use bincount, is about 7 times faster than np.unique
    # and orders of magnitude faster than pd.Series.value_counts
    counts = np.bincount(y, weights=w)
    non_zero = counts.nonzero()
    counts = counts[non_zero]
    proba = counts / w.sum()
    values = non_zero[0]

    return values, counts, proba

def entropy(proba):
    if len(proba) == 0:
        return 0
    return -np.sum(proba * np.log2(proba))

def majority_error(proba):
    if len(proba) == 0:
        return 0
    return 1 - proba.max()

def gini_index(proba):
    if len(proba) == 0:
        return 0
    return 1 - np.sum(proba ** 2)

def gain(X: np.ndarray, y: np.ndarray, w: np.ndarray, feature: int, feature_values: list, metric_func: callable):
    X = X[:,feature]
    n = w.sum()
    _, y_counts, y_proba = unique_proba(y, w)
    total_metric = metric_func(y_proba)
    w_metric_subsets = np.zeros(len(feature_values))

    for i, value in enumerate(feature_values):
        mask = X == value
        subset_labels = y[mask]
        subset_w = w[mask]
        n_subset = subset_w.sum()

        if n_subset > 0:
            _, _, subset_proba = unique_proba(subset_labels, subset_w)
            subset_metric = metric_func(subset_proba)
            w_metric_subsets[i] = subset_metric * (n_subset / n)

    return total_metric - np.sum(w_metric_subsets)

def avg_error(y_true, y_pred):
    return np.mean(y_true != y_pred)