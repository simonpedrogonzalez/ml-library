
import numpy as np
import pandas as pd
from utils import gain, entropy, majority_error, gini_index, argmax
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

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.labels = y.unique()
        self.features = X.columns.tolist()
        self.feature_values = {feature: X[feature].unique() for feature in self.features}
        self.tree = self._build_tree(self.X, self.y, self.features, 0)
        return self

    def _build_tree(self, X, y, features, depth):

        node = Node(ID3NodeData())

        label_counts = y.value_counts().to_dict()
        label_proba = y.value_counts(normalize=True).to_dict()
        majority_label = max(label_counts, key=label_counts.get)
        
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
            node.data.label = majority_label
            node.data.label_counts = label_counts
            node.data.label_proba = label_proba
            return node
        
        # Find best feature to split
        metric_value, feature_index, best_feature = self._pick_best_feature(X, y, features)

        node.data.next_feature = best_feature
        node.data.metric = { self.metric.__name__: metric_value } 
        
        # Get all possible values, not only the ones currently in X
        feature_values = self.feature_values[best_feature]
        
        # Add children
        for value in feature_values:
            subset = X[best_feature] == value
            X_subset = X[subset]

            if X_subset.empty:
                child = Node(ID3NodeData(
                    leaf_type = "empty_subset",
                    label = majority_label,
                    feature = best_feature,
                    value = value
                ))
                node.add_child(child)
                continue

            y_subset = y[subset]
            new_features = features.copy()
            new_features.pop(feature_index)

            child = self._build_tree(X_subset, y_subset, new_features, depth + 1)
            
            # add the data to the child node
            child.data.feature = best_feature
            child.data.value = value
            child.data.label_counts = y_subset.value_counts().to_dict()
            child.data.label_proba = y_subset.value_counts(normalize=True).to_dict()

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

    def _pick_best_feature(self, X, y, features):
        idx, minvalue = argmax([gain(X, y, feature, self.metric) for feature in features])
        return minvalue, idx, features[idx]
    

    