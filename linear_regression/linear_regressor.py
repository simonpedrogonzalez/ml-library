from abc import ABC, abstractmethod
import numpy as np

class LinearRegressor(ABC):

    def __init__(self, initial_weights='zeros'):
        self.initial_weights = initial_weights

    def fit(self, X: np.array , y: np.array):
        self.X = self._augment(X)
        self.X_T = self.X.T
        self.y = y
        if self.initial_weights == 'zeros':
            self.w = np.zeros(X.shape[1] + 1)
        self.w = self._fit()
        return self

    @abstractmethod
    def _fit(self):
        pass

    def _augment(self, X: np.array):
        return np.c_[np.ones(X.shape[0]), X]

    def predict(self, X: np.array):
        return np.dot(self._augment(X), self.w)

class AnalyticalRegressor(LinearRegressor):

    def _fit(self):
        X, y, X_T = self.X, self.y, self.X_T
        self.w = (X_T @ X).I @ X_T @ y
