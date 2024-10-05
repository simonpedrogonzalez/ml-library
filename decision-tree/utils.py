import numpy as np
import pandas as pd

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

@profile
def gain(X: pd.DataFrame, y: pd.Series, feature: str, metric_func: callable):
    X = X[feature].values
    y = y.values
    n = len(y)
    _, y_counts = np.unique(y, return_counts=True)
    y_proba = y_counts / n
    total_metric = metric_func(y_proba)
    feature_values = np.unique(X)
    w_metric_subsets = np.empty(len(feature_values))

    for i, value in enumerate(feature_values):
        subset_labels = y[X == value]
        n_subset = len(subset_labels)
        _, subset_counts = np.unique(subset_labels, return_counts=True)
        subset_proba = subset_counts / n_subset
        subset_metric = metric_func(subset_proba)
        subset_weight = n_subset / n
        w_metric_subsets[i] = subset_metric * subset_weight

    return total_metric - np.sum(w_metric_subsets)
    

def argmin(arr):
    return min(enumerate(arr), key=lambda x: x[1])

def argmax(arr):
    return max(enumerate(arr), key=lambda x: x[1])

def avg_error(y_true, y_pred):
    return np.mean(y_true != y_pred)
