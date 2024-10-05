import pandas as pd
import numpy as np

def transform_num_to_bin_median(X: pd.DataFrame):
    """
    Binarize numerical features in a DataFrame using median.
    """
    for feature in X.columns:
        if X[feature].dtype.kind in 'iufc': # integer, unsigned integer, float, complex
            median = X[feature].median()
            # binarize numerical features with the median
            leq_median_str = f' <={median:.2f}'
            gt_median_str = f' >{median:.2f}'
            X[feature] = np.where(X[feature] > median, gt_median_str, leq_median_str)
    return X

def impute_mode(X: pd.DataFrame, nan_value=None):
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

class CatEncodedSeries:
    '''Represent a pandas Series in an integer-encoded format, keeping track
    of the original categories and their integer codes. It makes sense to use
    it if other algorithm directly uses the mapping.
    '''

    def __init__(self):
        self.values = None
        self.categories = None
        self.category_index = None
        self.c2s = None
        self.s2c = None
    
    def from_pandas(self, series: pd.Series):
        self.values, self.categories, self.category_index, self.c2s, self.s2c = self._encode(series)
        return self

    def _encode(self, series: pd.Series):
        name = series.name
        s = series.astype('category')
        categories = s.cat.categories
        c2s = {i: v for i, v in enumerate(categories)}
        s2c = {v: i for i, v in enumerate(categories)}
        return s.cat.codes.values, categories, list(range(len(categories))), c2s, s2c

class CatEncodedDataFrame:
    '''Represent a pandas DataFrame in an integer-encoded format, keeping track
    of the original categories and their integer codes. It makes sense to use
    it if other algorithm directly uses the mapping, or just to transform the
    data to a more efficient format while keeping track of the original categories.
    '''

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
