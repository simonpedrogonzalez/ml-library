import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from copy import deepcopy
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
import pandas as pd
import numpy as np

class AdaBoost:
    def __init__(self, learner, n_learners):
        self.learner = learner
        self.n_learners = n_learners
        self.learners = [deepcopy(self.learner) for _ in range(self.n_learners)]
        self.alphas = np.zeros(self.n_learners)

    def _preprocess(self, data, labels):
        if isinstance(data, pd.DataFrame):
            data = CatEncodedDataFrame().from_pandas(data)
        if not isinstance(data, CatEncodedDataFrame):
            raise ValueError('Invalid data type')
        self.data = data
        self.X = data.X
        self.features = data.features
        self.feature_index = data.feature_index
        self.feature_values = data.feature_values
        self.c2s = data.c2s
        self.s2c = data.s2c
        if isinstance(labels, pd.Series):
            labels = CatEncodedSeries().from_pandas(labels)
        if not isinstance(labels, CatEncodedSeries):
            raise ValueError('Invalid labels type')
        self.labels = labels
        self.y = labels.values
        self.label_values = labels.categories
        self.label_index = labels.category_index
        self.lc2s = labels.c2s
        self.ls2c = labels.s2c

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._preprocess(X, y)
        self._build_ensemble(self.X, self.y)
        return self

    def _build_ensemble(self, X: np.ndarray, y: np.ndarray):

        n = X.shape[0]
        weights = np.ones(X.shape[0]) / n
        
        for i in range(self.n_learners):
            # print(f"Training learner {i+1}/{self.n_learners}")
            # print("weights", weights*len(weights))
            self.learners[i].fit(self.data, self.labels, sample_weight=weights)
            y_pred = self.learners[i].predict(X)
            error = np.sum(weights * (y_pred != y))
            # print(f"Error: {error}")
            alpha = 0.5 * np.log((1 - error) / error)
            weights = self._update_weights(weights, y, y_pred, alpha)
            # print("weights", weights)
            self.alphas[i] = alpha

        return self
        
    def _update_weights(self, w, y, y_pred, alpha):
        # If there are more than 2 classes, we
        # can't rely on y_pred*y to get the
        # error = -1 and ok = 1
        comparison = 2 * (y_pred == y).astype(int) - 1
        new_w = w * np.exp(-alpha * comparison)
        return new_w / np.sum(new_w)
    
    def predict(self, X: pd.DataFrame):
        return_str_values = False
        
        if isinstance(X, pd.DataFrame):
            return_str_values = True
            X = CatEncodedDataFrame().from_pandas(X).X
        if isinstance(X, CatEncodedDataFrame):
            X = X.X
        if not isinstance(X, np.ndarray):
            raise ValueError('Invalid data type')
        
        y_preds = np.empty((X.shape[0], self.n_learners), dtype=int)
        
        for i, learner in enumerate(self.learners):
            y_preds[:, i] = learner.predict(X)

        y_pred_votes = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(self.label_values), weights=self.alphas),
            axis=1, arr=y_preds)
        
        final_preds = np.argmax(y_pred_votes, axis=1)

        if return_str_values:
            return pd.Series(np.vectorize(self.lc2s.get)(final_preds))

        return final_preds



