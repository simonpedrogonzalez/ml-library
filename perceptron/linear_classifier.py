import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
from abc import ABC, abstractmethod

class LinearClassifier(ABC):

    def __init__(self, initial_weights=0):
        self.initial_weights = initial_weights

    def fit(self, X: np.array , y: np.array):
        self.X = self._augment(X)
        self.X_T = self.X.T
        self.y = y
        if isinstance(self.initial_weights, int):
            self.w = np.ones(self.X.shape[1]) * self.initial_weights
        else:
            self.w = self.initial_weights
        self._fit()
        return self

    @abstractmethod
    def _fit(self):
        pass

    def _augment(self, X: np.array):
        return np.c_[np.ones(X.shape[0]), X]

    def predict(self, X: np.array):
        return np.sign(np.dot(self._augment(X), self.w))

