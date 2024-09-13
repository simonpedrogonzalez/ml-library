import numpy as np
import pandas as pd

def entropy(y: pd.Series):
    proba = y.value_counts(normalize=True)
    return -np.sum(proba * np.log2(proba))

def information_gain(X: pd.DataFrame, y: pd.Series, feature: str):
    total_entropy = entropy(y)
    feature_values = X[feature].unique()
    weights = X[feature].value_counts(normalize=True)
    subset_entropies = []
    for value in feature_values:
        subset = y[X[feature] == value]
        subset_entropy = entropy(subset)
        subset_entropies.append(subset_entropy)
    return total_entropy - np.sum(weights * np.array(subset_entropies))

def majority_error(y: pd.Series):
    return 1 - y.value_counts(normalize=True).max()

def gini_index(y: pd.Series):
    proba = y.value_counts(normalize=True)
    return 1 - np.sum(proba ** 2)

def argmin_min(arr):
    return min(enumerate(arr), key=lambda x: x[1])

def argmax_max(arr):
    return max(enumerate(arr), key=lambda x: x[1])