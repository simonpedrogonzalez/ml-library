# import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from copy import deepcopy
import pandas as pd
import numpy as np

class AdaBoost:
    def __init__(self, learner, n_learners):
        self.learner = learner
        self.n_learners = n_learners
        self.learners = [deepcopy(self.learner) for _ in range(self.n_learners)]
        self.alphas = np.zeros(self.n_learners)

    def fit(self, X, y):
        weights = np.ones(X.shape[0]) / X.shape[0]

        for i in range(self.n_learners):
            self.learners[i].fit(X, y, sample_weight=weights)
            y_pred = self.learners[i].predict(X)
            error = np.sum(weights * (predictions != y))
            alpha = 0.5 * np.log((1 - error) / error)
            weights = self._update_weights(weights, y, y_pred, alpha)
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
        
        predictions = np.zeros((X.shape[0], self.n_learners))
        for i, learner in enumerate(self.learners):
            predictions[:, i] = learner.predict(X)
        
        final_predictions = np.sum(predictions*alphas, axis=1).sign()

        return final_predictions



