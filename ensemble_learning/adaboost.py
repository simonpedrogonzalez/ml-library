import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from copy import deepcopy
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
import pandas as pd
import numpy as np

class AdaBoost:
    def __init__(self, learner, n_learners):
        self.learner = learner
        self.initial_n_learners = n_learners
        self.n_learners = 0
        self.learners = []
        self.alphas = []
        self.predictions = []

    def _preprocess(self, data, labels):
        # self.X = data.X
        self.X = data.X
        # self.features = data.features
        # self.feature_index = data.feature_index
        # self.feature_values = data.feature_values
        # self.c2s = data.c2s
        # self.s2c = data.s2c
        # self.labels = labels
        # self.y = labels.values
        self.y = labels.values
        self.label_values = labels.categories
        # self.label_index = labels.category_index
        # self.lc2s = labels.c2s
        # self.ls2c = labels.s2c

    def fit(self, X: CatEncodedDataFrame, y: CatEncodedSeries):
        self._preprocess(X, y)
        self._build_ensemble(self.X, self.y)
        return self

    def _build_ensemble(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        self.w = np.ones(n) / n
        
        for i in range(self.initial_n_learners):
            self.fit_new_learner()
        
        return self
    
    def fit_new_learner(self):
        learner = deepcopy(self.learner)
        learner.fit(self.X, self.y, sample_weight=self.w)
        y_pred = learner.predict(self.X)
        error = np.sum(self.w * (y_pred != self.y))
        alpha = 0.5 * np.log((1 - error) / error)
        self.w = self._update_weights(self.w, self.y, y_pred, alpha)
        self.alphas.append(alpha)
        # print(f"Alpha: {np.round(self.alphas, 2)}")
        self.learners.append(learner)
        self.n_learners += 1
        return self
        
    def _update_weights(self, w, y, y_pred, alpha):
        # If there are more than 2 classes, we
        # can't rely on y_pred*y to get the
        # error = -1 and ok = 1
        comparison = 2 * (y_pred == y).astype(int) - 1
        new_w = w * np.exp(-alpha * comparison)
        new_w = new_w / np.sum(new_w)
        failed = y_pred != y
        failed_indices = np.where(failed)[0]
        failed_w = new_w[failed_indices]
        # print(f"Failed: {failed_indices}, old_w: {np.round(w[failed_indices], 2)}, new_w: {np.round(failed_w, 2)}")
        return new_w
        
    
    def _get_voted_predictions(self, learners, alphas, X):
        y_preds = np.empty((X.shape[0], len(learners)), dtype=int)
        alphas = np.array(alphas)
        for i, learner in enumerate(learners):
            y_preds[:, i] = learner.predict(X)
        voted_y_preds = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(self.label_values), weights=alphas),
            axis=1, arr=y_preds)
        return voted_y_preds
    
    def predict(self, X: CatEncodedDataFrame):
        X = X.X

        voted_preds = self._get_voted_predictions(self.learners, self.alphas, X)
        final_preds = np.argmax(voted_preds, axis=1)

        self.predictions.append((X, voted_preds, self.n_learners))
        return final_preds

    def re_predict(self, prediction_index):
        """Get a new prediction by adding the predictions of the new learners only
        """
        X, old_voted_preds, old_n_learners = self.predictions[prediction_index]
        new_learners = self.learners[old_n_learners:]
        new_alphas = self.alphas[old_n_learners:]
        new_voted_preds = self._get_voted_predictions(new_learners, new_alphas, X)
        voted_preds = old_voted_preds + new_voted_preds
        final_preds = np.argmax(voted_preds, axis=1)
        self.predictions[prediction_index] = (X, voted_preds, self.n_learners)
        return final_preds

