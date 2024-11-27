import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
from perceptron.linear_classifier import LinearClassifier
from utils.stats import endless_batch_generator
from scipy.optimize import minimize


class LearningRateSchedule:
    def __init__(self, function, params):
        self.function = function
        self.params = params

    def __call__(self, *args):
        return self.function(*args, **self.params)

class PrimalSGDSVM(LinearClassifier):

    def __init__(self, lr_schedule: LearningRateSchedule, C=1, max_epochs=100, atol=1e-5, initial_weights=0):
        self.C = C
        self.lr_schedule = lr_schedule
        self.max_epochs = max_epochs
        self.atol = atol
        self.batch_generator = None
        self.epoch = None
        self.n = None
        self.last_w = None
        self.n_iter = None
        super().__init__(initial_weights)

    def should_stop(self):
        return self.epoch >= self.max_epochs # \
            # or self.converged() # not asked by the exercise but it's a good idea to find best hyperparameters
    
    def _weight_difference(self):
        return np.linalg.norm(self.w - self.last_w)

    def converged(self):
        return self._weight_difference() < self.atol

    def _fit(self):
        self.n_iter = 0
        self.batch_generator = endless_batch_generator(
            self.X, self.y, 1, random=True, return_epoch=True
        )
        self.epoch = 0
        self.n = self.X.shape[0]

        while True:
            self.step()
            if self.should_stop():
                break
        
        return self

    def step(self):
        self.epoch, (X, y) = next(self.batch_generator)
        X, y = X[0], y[0] # since is just 1
        self.last_w = self.w.copy()
        w= self.w
        lr = self.lr_schedule(self.n_iter)
        y_pred = w @ X
        if y_pred * y <= 1:
            self.w = w - lr * w + lr * self.C * self.n * y * X
        else:
            self.w = (1 - lr) * w

        if np.isnan(self.w).any():
            # Obviously diverged
            raise ValueError("Learning diverged, learning rate might be too high")

        self.n_iter += 1 # keep count of the iters
        return self


class DualSVM(LinearClassifier):

    def __init__(self, C=1):
        self.C = C
        self.alphas = None
        super().__init__(0)

    def _fit(self):
        self.alphas = np.zeros(self.X.shape[0])
        self.gram_matrix = self.X @ self.X.T
        self.yiyj = self.y * self.y[:, None]

        def objective(alphas):
            aiaj = alphas[:, None] * alphas
            # since objective is defined inside _fit, we can access self
            return 0.5 * np.sum(aiaj * self.yiyj * self.gram_matrix) - np.sum(alphas)
        
        # equality constraint
        constraints = [{'type': 'eq', 'fun': lambda a: np.sum(a * self.y)}]

        # bounds on alphas
        bounds = [(0, self.C) for _ in range(self.X.shape[0])]

        result = minimize(fun=objective, x0=self.alphas, constraints=constraints, bounds=bounds, method='SLSQP')    

        self.alphas = result.x
        self.w = np.sum(self.alphas[:, None] * self.y[:, None] * self.X, axis=0)
        return self


class Kernel:
    def __init__(self, function, params):
        self.params = params
        self.function = function

    def __call__(self, X, Z):
        return self.function(X, Z, **self.params)

class KernelSVM(LinearClassifier):

    def __init__(self, C=1, kernel=None):
        self.C = C
        self.kernel = kernel
        self.alphas = None
        super().__init__(0)
    
    def _fit(self):

        self.alphas = np.zeros(self.X.shape[0])
        self.gram_matrix = self.kernel(self.X, self.X)
        self.yiyj = self.y * self.y[:, None]

        def objective(alphas):
            aiaj = alphas[:, None] * alphas
            # since objective is defined inside _fit, we can access self
            return 0.5 * np.sum(aiaj * self.yiyj * self.gram_matrix) - np.sum(alphas)

        # equality constraint
        constraints = [{'type': 'eq', 'fun': lambda a: np.sum(a * self.y)}]

        # bounds on alphas
        bounds = [(0, self.C) for _ in range(self.X.shape[0])]

        result = minimize(fun=objective, x0=self.alphas, constraints=constraints, bounds=bounds, method='SLSQP')

        self.alphas = result.x
        self.w = np.sum(self.alphas[:, None] * self.y[:, None] * self.X, axis=0)
        return self

    def predict(self, X):
        X_ = self._augment(X)
        K = self.kernel(self.X, X_)
        return np.sign(K.T @ (self.alphas * self.y))