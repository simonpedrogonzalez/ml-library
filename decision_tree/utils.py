import numpy as np
import pandas as pd

def unique_proba(y: np.ndarray):
    # assume int array and use bincount, is about 7 times faster than np.unique
    # and orders of magnitude faster than pd.Series.value_counts
    counts = np.bincount(y)
    non_zero = counts.nonzero()
    counts = counts[non_zero]
    proba = counts / len(y)
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

def gain(X: np.ndarray, y: np.ndarray, feature: int, feature_values: list, metric_func: callable):
    X = X[:,feature]
    n = len(y)
    _, y_counts, y_proba = unique_proba(y)
    total_metric = metric_func(y_proba)
    w_metric_subsets = np.zeros(len(feature_values))

    for i, value in enumerate(feature_values):
        subset_labels = y[X == value]
        n_subset = len(subset_labels)
        if n_subset > 0:
            _, _, subset_proba = unique_proba(subset_labels)
            subset_metric = metric_func(subset_proba)
            subset_weight = n_subset / n
            w_metric_subsets[i] = subset_metric * subset_weight

    return total_metric - np.sum(w_metric_subsets)

def avg_error(y_true, y_pred):
    return np.mean(y_true != y_pred)
