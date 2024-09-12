
import numpy as np
from utils import entropy, information_gain, majority_error, gini_index, min_argmin
from tree import Tree, Node

class ID3:
    def __init__(self, metric, max_depth):
        self.metric = metric
        self.max_depth = max_depth
        self.metric_func = {
            'infogain': self.information_gain,
            'majerr': self.majority_error,
            'gini': self.gini_index
        }.get(metric)
        if self.metric_func is None:
            raise ValueError('Invalid metric')

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.labels = y.unique()
        self._build_tree()

    def _build_tree(self):
        features = self.X.columns
        self.tree = Tree(Node())
        _build_tree_recursive(self.X, features, self.tree.root)

    def _build_tree_recursive(self, X, y, remaining_features, node):
        # Base case no remaining features
        if len(remaining_features) == 0:
            majority_label = y.value_counts().idxmax()
            node.value = majority_label
            return

        # Base case all labels are the same
        if len(np.unique(y)) == 1:
            return

        # Base case max depth reached
        if self.tree.get_depth() == self.max_depth:
            return
        
        # Find best feature to split
        best_feature = self.best_split(X, y, remaining_features)
        feature_values = X[best_feature].unique()
        
        # Add children
        for value in feature_values:
            child = Node(feature=best_feature, value=value)
            node.add_child(child)
            subset = X[best_feature] == value
            X_subset = X[subset]
            y_subset = y[subset]
            remaining_features = remaining_features.drop(best_feature)
            self._build_tree_recursive(X_subset, y_subset, remaining_features, child)

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self._predict_recursive(row, self.tree.root))
        
    def _predict_recursive(self, row, node):
        if node.is_leaf():
            return node.value
        for child in node.children:
            if row[node.feature] == child.value:
                return self._predict_recursive(row, child)
        return None

    def best_split(self, X, y, features):
        best_feature = None
        best_score = -np.inf
        for feature in features:
            score = self.metric_func(X, y, feature)
            if score > best_score:
                best_score = score
                best_feature = feature
        return best_feature

    def pick_best_feature(self, X, y, features):
        _, argmin = min_argmin([self.metric_func(X, y, feature) for feature in features])
        return features[argmin]
    

    