import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from utils.stats import endless_batch_generator
from perceptron.linear_classifier import LinearClassifier
import numpy as np
from abc import ABC, abstractmethod


class Perceptron(LinearClassifier):

    def __init__(self, lr=0.01, max_epochs=10, initial_weights=0):
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_generator = None
        self.n_errors = None
        self.epoch = None
        super().__init__(initial_weights)

    def should_stop(self):
        return self.since_last_error >= self.X.shape[0] or \
        self.epoch >= self.max_epochs

    def _fit(self):
        self.n_iter = 0
        self.n_errors = None
        self.batch_generator = endless_batch_generator(
            self.X, self.y, 1, random=True, return_epoch=True
        )
        self.epoch = 0
        n = self.X.shape[0]
        self.since_last_error = 0

        while True:
            self.step()
            if self.should_stop():
                break
        
        return self
    
    def step(self):
        self.epoch, (X, y) = next(self.batch_generator)
        X, y = X[0], y[0] # since is just 1 sample
        w, lr = self.w, self.lr
        y_pred = w @ X
        if y_pred * y <= 0:
            self.w = w + lr * y * X
            self.since_last_error = 0
        else:
            self.since_last_error += 1

        if np.isnan(self.w).any():
             # Obviously diverged
            raise ValueError("Learning rate too high")

        self.n_iter += 1 # keep count of the iters
        return self