import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np

class Loss:
    def __init__(self, function, prime):
        self.function = function
        self.prime = prime
    def __call__(self, y_true, y_pred):
        return self.function(y_true, y_pred)
    def prime(self, y_true, y_pred):
        return self.prime(y_true, y_pred)


def fix_inputs(y_true, y_pred):
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    
    # make sure they have the same shape, if not, transpose one of them
    if y_true.shape[0] != y_pred.shape[0]:
        y_true = y_true.T
    return y_true, y_pred


def cel(y_true, y_pred):
    y_true, y_pred = fix_inputs(y_true, y_pred)
    res = -np.mean(y_true * np.log(y_pred + 1e-12) + (1 - y_true) * np.log(1 - y_pred + 1e-12))
    if res.size == 1 and res.shape != ():
        return res.flatten()[0]
    return res

def cel_prime(y_true, y_pred):
    y_true, y_pred = fix_inputs(y_true, y_pred)
    res = -(y_true / (y_pred + 1e-12) - (1 - y_true) / (1 - y_pred + 1e-12))
    if res.size == 1 and res.shape != ():
        return res.flatten()[0]
    return res


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__(cel, cel_prime)


class LearningRateSchedule:
    def __init__(self, function, params):
        self.function = function
        self.params = params

    def __call__(self, *args):
        return self.function(*args, **self.params)


class TimeBasedDecay(LearningRateSchedule):
    def __init__(self, gamma_0, d):
        f = lambda t, gamma_0, d: gamma_0 / (1 + gamma_0 / d * t)
        super().__init__(f, {'gamma_0': gamma_0, 'd': d})


class Activation:
    def __init__(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime

    def __call__(self, z):
        return self.f(z)

    def prime(self, z):
        return self.f_prime(z)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


class Layer:

    def __init__(self, input_size, output_size, f=lambda x: x):
        self.input_size = input_size 
        self.output_size = output_size
        self.w = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.f = f

    def forward(self, x):
        self.x = np.atleast_2d(x)
        self.z = x @ self.w + self.b
        return self.f(self.z)

    def backward(self, grad):
        grad = grad * self.f.prime(self.z)
        self.grad_w = self.x.T @ grad
        self.grad_b = np.sum(grad, axis=0, keepdims=True)
        self.grad = grad @ self.w.T
        # print(self.grad.shape)
        return self.grad, self.grad_w, self.grad_b

    def update(self, lr):
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b

class NN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad, grad_w, grad_b = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)