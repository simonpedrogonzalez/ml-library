import numpy as np
import pandas as pd

# @profile
def label_proba(y):
    _, counts = np.unique(y, return_counts=True)
    proba = counts / len(y)
    return proba

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

# @profile
def gain(X: pd.DataFrame, y: pd.Series, feature: str, metric_func: callable):
    X = X[feature].values
    y = y.values
    total_metric = metric_func(label_proba(y))
    feature_values = np.unique(X)
    w_metric_subsets = np.empty(len(feature_values))
    total = len(y)

    for i, value in enumerate(feature_values):
        subset_labels = y[X == value]
        subset_proba = label_proba(subset_labels)
        subset_metric = metric_func(subset_proba)
        subset_weight = len(subset_labels) / total
        w_metric_subsets[i] = subset_metric * subset_weight

    return total_metric - np.sum(w_metric_subsets)
    

def argmin(arr):
    return min(enumerate(arr), key=lambda x: x[1])

def argmax(arr):
    return max(enumerate(arr), key=lambda x: x[1])

def avg_error(y_true, y_pred):
    return np.mean(y_true != y_pred)
