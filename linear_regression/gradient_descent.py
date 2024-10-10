import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from linear_regression.linear_regressor import LinearRegressor
from utils.stats import sample

class GradientDescent(LinearRegressor):

    def __init__(self, lr=0.01, ntol=1e-6, n_iter=1000):
        self.lr = lr
        self.ntol = ntol
        self.n_iter = n_iter
        super().__init__()
    
    def step(self):
        self.w = self._step()
        return self
    
    @abstractmethod
    def _step(self) -> np.array:
        pass
    
    def _should_stop(self, i, w, new_w):
        # Stop with max number of iterations also just in case
        return i >= self.n_iter or np.linalg.norm(new_w - w) < self.ntol

    def _fit(self):
        w = self.w
        for i in range(self.n_iter):
            self.step()
            if self._should_stop(i, w, new_w):
                break
            w = new_w

class StochasticGradientDescent(GradientDescent):

    def _step(self):
        X, y, w, lr = self.X, self.y, self.w, self.lr
        for i in range(X.shape[0]):
            y_pred = X[i] @ w
            error = y_pred - y[i]
            w = w + lr * X[i] * error
        return w
    
class BatchGradientDescent(GradientDescent):

    def __init__(self, batch_size, *args, **kwargs):
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)

    def _get_batches(self):
        X, y, n, size = self.X, self.y, self.X.shape[0], self.batch_size
        n_batches = n // size
        indices = np.random.shuffle(np.arange(n))
        for i in range(n_batches):
            if 
            batch = indices[i*size:(i+1)*size]
            yield X[batch], y[batch]

        for i in range(n_batches):
            batch = sample(size, X, y, replace=False)

    def _step(self):
        X, y, w, lr = self.X, self.y, self.w, self.learning_rate
        for X_batch, y_batch in self._get_batches():
            y_pred = X_batch @ w
            error = y_pred - y_batch
            w = w - lr * X_batch.T @ error
        return w