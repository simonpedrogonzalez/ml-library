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
