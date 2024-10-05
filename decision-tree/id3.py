
import numpy as np
import pandas as pd
from utils import gain, entropy, majority_error, gini_index, cat_series_to_np, cat_df_to_np, unique_proba
from tree import Node, ID3NodeData

class ID3:
    def __init__(self, metric, max_depth=None):
        self.metric = metric
        self.max_depth = max_depth
        self.metric = {
            'infogain': entropy,
            'majerr': majority_error,
            'gini': gini_index
        }.get(metric)
        if self.metric is None:
            raise ValueError('Invalid metric')
        self.tree = None

    # @profile
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X, self.features, self.feature_index, \
            self.feature_values, self.c2s, self.s2c = cat_df_to_np(X)
        self.y, self.label_values, self.label_index, \
            self.lc2s, self.ls2c = cat_series_to_np(y)
        # self.preprocess(X, y)
        # self.X = X
        # self.y = y
        # self.labels = y.unique()
        # self.features = X.columns.tolist()
        # self.feature_values = {feature: X[feature].unique() for feature in self.features}
        self.tree = self._build_tree(self.X, self.y, self.feature_index, 0)
        return self

    # @profile
    def _build_tree(self, X: np.ndarray, y:np.ndarray, features:list, depth:int):

        node = Node(ID3NodeData())

        label_values, label_counts, label_proba = unique_proba(y)
        majority_label =  label_values[np.argmax(label_counts)]
        majority_label_name = self.lc2s[majority_label]

        # Base cases
        base_case = None
        if len(features) == 0:
            base_case = "no_features"
        if len(label_counts) == 1:
            base_case = "pure_label"
        if self.max_depth and depth >= self.max_depth:
            base_case = "max_depth"

        if base_case is not None:
            node.data.leaf_type = base_case
            node.data.label = majority_label_name
            node.data.label_counts = {k: v for k, v in zip(label_values, label_counts)}
            node.data.label_proba = {k: v for k, v in zip(label_values, label_proba)}
            return node

        # Find best feature to split
        metric_value, feature_index, best_feature = self._pick_best_feature(X, y, features)
        best_feature_name = self.features[best_feature]

        node.data.next_feature = best_feature_name
        node.data.metric = { self.metric.__name__: metric_value }

        # Get all possible values, not only the ones currently in X
        feature_values = self.feature_values[best_feature]

        # Add children
        for value in feature_values:
            value_name = self.c2s[(best_feature_name, value)]

            # if best_feature_name == 'doors':
            #     print('here')

            # if value_name == '5more':
            #     print('here')

            subset = X[:, best_feature] == value
            X_subset = X[subset]

            if X_subset.size == 0:
                child = Node(ID3NodeData(
                    leaf_type = "empty_subset",
                    label = majority_label_name,
                    feature = best_feature_name,
                    value = value_name
                ))
                node.add_child(child)
                continue

            y_subset = y[subset]
            new_features = features.copy()
            new_features.pop(feature_index)

            child = self._build_tree(X_subset, y_subset, new_features, depth + 1)

            # add the data to the child node
            child.data.feature = best_feature_name
            child.data.value = value_name
            child_label_values, child_label_counts, child_label_proba = unique_proba(y_subset)
            child.data.label_counts = {k: v for k, v in zip(child_label_values, child_label_counts)}
            child.data.label_proba = {k: v for k, v in zip(child_label_values, child_label_proba)}

            node.add_child(child)

        return node

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self._predict(row, self.tree))
        return pd.Series(predictions)

    def _predict(self, row, node):
        if node.is_leaf():
            return node.data.label
        feature = node.data.next_feature
        value = row[feature]
        for child in node.children:
            if child.data.value == value:
                return self._predict(row, child)
        raise ValueError(f"Value {value} not found")

    # @profile
    def _pick_best_feature(self, X, y, features):
        gains = np.array([
            gain(X, y, feature, self.feature_values[feature], self.metric) \
                for feature in features
            ])
        idx = np.argmax(gains)
        minvalue = gains[idx]
        return minvalue, idx, features[idx]


