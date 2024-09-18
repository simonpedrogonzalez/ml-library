import numpy as np
import pandas as pd

def label_proba(y: pd.Series, w: pd.Series = None):
    if w is None:
        return y.value_counts(normalize=True)
    return w.groupby(y).sum() / w.sum()

def entropy(proba: pd.Series):
    if proba.empty:
        return 0
    return -np.sum(proba * np.log2(proba))

def majority_error(proba: pd.Series):
    if proba.empty:
        return 0
    return 1 - proba.max()

def gini_index(proba: pd.Series):
    if proba.empty:
        return 0
    return 1 - np.sum(proba ** 2)

def gain(X: pd.DataFrame, y: pd.Series, feature: str, metric_func: callable, w: pd.Series = None):

    if w is None:
        w = pd.Series(np.ones(len(y)), index=y.index)

    l_proba = label_proba(y, w)
    
    total_metric = metric_func(l_proba)
    feature_values = X[feature].unique()
    w_metric_subsets = []

    for value in feature_values:
        subset_indices = X[feature] == value
        subset_labels = y[subset_indices]
        subset_w = w[subset_indices]
        subset_proba = label_proba(subset_labels, subset_w)
        subset_metric = metric_func(subset_proba)
        subset_weight = subset_w.sum() / w.sum()
        w_metric_subsets.append(subset_metric * subset_weight)

    return total_metric - np.sum(w_metric_subsets)

def argmin(arr):
    return min(enumerate(arr), key=lambda x: x[1])

def argmax(arr):
    return max(enumerate(arr), key=lambda x: x[1])

def avg_error(y_true, y_pred):
    return np.mean(y_true != y_pred)
