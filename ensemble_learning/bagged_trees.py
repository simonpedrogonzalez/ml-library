import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
from copy import deepcopy
import numpy as np


class BaggedTrees:
    def __init__(self, tree, n_trees):
        self.tree = tree
        self.initial_n_trees = n_trees
        self.n_trees = 0
        self.trees = []
        self.predictions = []

    def _preprocess(self, data, labels):
        self.X = data.X
        self.y = labels.values
        self.label_values = labels.categories
        
    def fit(self, X: CatEncodedDataFrame, y: CatEncodedSeries):
        self._preprocess(X, y)
        self._build_ensemble(self.X, self.y)
        return self

    def _build_ensemble(self, X, y):
        for i in range(self.initial_n_trees):
            self.fit_new_tree()
        return self

    def sample_with_replacement(self, X, y):
        # Its a boostrap sample, so we take the same number of samples
        idx = np.random.choice(range(len(X)), size=len(X), replace=True)
        return X[idx], y[idx]

    def fit_new_tree(self):
        X, y = self.sample_with_replacement(self.X, self.y)
        tree = deepcopy(self.tree)
        tree.fit(X, y)
        self.trees.append(tree)
        self.n_trees += 1
        return self

    def _get_voted_predictions(self, trees, X):
        y_preds = np.empty((X.shape[0], len(trees)), dtype=int)
        for i, tree in enumerate(trees):
            y_preds[:, i] = tree.predict(X)
        voted_y_pred = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(self.label_values)),
            axis=1, arr=y_preds)
        return voted_y_pred
            
    def predict(self, X: CatEncodedDataFrame):
        X = X.X

        voted_preds = self._get_voted_predictions(self.trees, X)
        final_preds = np.argmax(voted_preds, axis=1)

        self.predictions.append((X, voted_preds, self.n_trees))
        return final_preds

    def re_predict(self, prediction_index):
        """Get a new prediction by adding the predictions of the new tree only
        """
        X, old_voted_preds, old_n_trees = self.predictions[prediction_index]
        new_trees = self.trees[old_n_trees:]
        new_voted_preds = self._get_voted_predictions(new_trees, X)
        voted_preds = old_voted_preds + new_voted_preds
        final_preds = np.argmax(voted_preds, axis=1)
        self.predictions[prediction_index] = (X, voted_preds, self.n_trees)
        return final_preds