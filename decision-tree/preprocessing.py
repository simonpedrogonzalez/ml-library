import pandas as pd
import numpy as np

def transform_num_to_bin_median(X):
    """
    Binarize numerical features in a DataFrame using median.
    """
    for feature in X.columns:
        if X[feature].dtype.kind in 'iufc': # integer, unsigned integer, float, complex
            median = X[feature].median()
            # binarize numerical features with the median
            X[feature] = np.where(X[feature] > median, '>Me', '<=Me')
    return X

def impute_mode(X, nan_value=None):
    """
    Impute missing values with the most common value in each column.
    Does not work for continuous numerical features.
    """

    if nan_value is None:
        nan_value = np.nan

    for feature in X.columns:
        values = X[feature].unique()
        if not nan_value in values:
            continue
        if len(values) == 1:
            raise ValueError(f"Feature '{feature}' has only {nan_value} values.")

        mode, second_mode = X[feature].value_counts().index[:2].to_list()
        if mode == nan_value:
            mode = second_mode

        X[feature] = X[feature].replace(nan_value, mode)
    
    return X