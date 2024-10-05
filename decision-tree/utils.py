import numpy as np
import pandas as pd

# @profile
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

# @profile
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
    
def argmin(arr):
    return min(enumerate(arr), key=lambda x: x[1])

def argmax(arr):
    return max(enumerate(arr), key=lambda x: x[1])

def avg_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

class CatEncodedSeries:

    def __init__(self):
        self.series = None
        self.categories = None
        self.category_index = None
        self.c2s = None
        self.s2c = None
    
    def from_pandas(self, series: pd.Series):
        self.series, self.categories, self.category_index, self.c2s, self.s2c = self._encode(series)
        return self

    def _encode(self, series: pd.Series):
        name = series.name
        s = series.astype('category')
        categories = s.cat.categories
        c2s = {i: v for i, v in enumerate(categories)}
        s2c = {v: i for i, v in enumerate(categories)}
        return s.cat.codes.values, categories, list(range(len(categories))), c2s, s2c



class CatEncodedDataFrame:
    def __init__(self):
        self.X = None
        self.features = None
        self.feature_index = None
        self.feature_values = None
        self.c2s = None
        self.s2c = None
    
    def from_pandas(self, X: pd.DataFrame):
        self.X, self.features, self.feature_index, self.feature_values, self.c2s, self.s2c = self._encode(X)
        return self

    def to_pandas(self):
        decoded_X = pd.DataFrame(self.X, columns=self.features)
        for name, col in decoded_X.items():
            decoded_X[name] = decoded_X[name].apply(lambda x: self.c2s[(name, x)])
        return decoded_X

    def _encode(self, X: pd.DataFrame):
        features = X.columns.tolist()
        feature_index = list(range(len(features)))
        feature_values = {}
        c2s = {}
        s2c = {}
        encoded_X = X.copy()
        encoded_X = X.astype('category')
        for i, (name, col) in enumerate(encoded_X.items()):
            feature_values[i] = []
            for j, v in enumerate(col.cat.categories):
                c2s[(name, j)] = v
                s2c[(name, v)] = j
                feature_values[i].append(j)
            encoded_X[name] = col.cat.codes
        encoded_X = encoded_X.values
        # test_cat_df_to_np(encoded_X, features, c2s)
        return encoded_X, features, feature_index, feature_values, c2s, s2c


    def _decode(self, encoded_X, features, c2s):
        decoded_X = pd.DataFrame(encoded_X, columns=features)
        for name, col in decoded_X.items():
            decoded_X[name] = decoded_X[name].apply(lambda x: c2s[(name, x)])
        return decoded_X
        
    def _test(self, X, decoded_X):
        assert X.equals(decoded_X)