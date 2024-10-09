import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
from utils.stats import bootstrap_sample
from ensemble_learning.ensemble import Ensemble
import numpy as np

class BaggedTrees(Ensemble):

    def fit_new_learner(self):
        X, y = bootstrap_sample(self.X, self.y)
        tree = self.learner_model.copy()
        tree.fit(X, y)
        self.trained_learners.append(tree)
        return self

    def _get_voted_predictions(self, learners_idx: list, X: np.ndarray):
        learners = [self.trained_learners[i] for i in learners_idx]
        y_preds = np.empty((X.shape[0], len(learners)), dtype=int)
        for i, tree in enumerate(learners):
            y_preds[:, i] = tree.predict(X)
        # Get a matrix with the count of votes for each label class
        voted_y_pred = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(self.label_values)),
            axis=1, arr=y_preds)
        return voted_y_pred
