import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import pandas as pd
import numpy as np
from time import time
from data.datasets import bank_note_dataset
from neural_networks.nn import NN, Layer, Sigmoid, CrossEntropyLoss, TimeBasedDecay
from utils.stats import endless_batch_generator, avg_error
import seaborn as sns
import matplotlib.pyplot as plt

def train(model, X, y, lr_schedule, loss, n_iter):
    y_hat = model.forward(X)
    l = loss(y, y_hat)
    lr = lr_schedule(n_iter)
    grad = loss.prime(y, y_hat)
    model.backward(grad)
    model.update(lr)

def test(model, X, y, loss):

    y_hat = model.forward(X)
    return loss(y, y_hat)

def log(epoch, n_iter, train_error, test_error):
    print(f"Epoch {epoch}, Iter {n_iter}, Train error: {train_error}, Test error: {test_error}")


def run(width):
    # Load data
    data = bank_note_dataset().to_numpy()

    data.train_labels = (data.train_labels == 1).astype(int)
    data.test_labels = (data.test_labels == 1).astype(int)

    max_epochs = 100
    lr_schedule = TimeBasedDecay(gamma_0=0.1, d=100)
    loss = CrossEntropyLoss()

    hidden_size = width
    n_hidden = 1
    output_size = 1
    input_size = data.train.shape[1]

    model = NN([
        Layer(input_size, hidden_size, Sigmoid()),
        *[Layer(hidden_size, hidden_size, Sigmoid()) for _ in range(n_hidden)],
        Layer(hidden_size, output_size, Sigmoid())
    ])

    batch_generator = endless_batch_generator(data.train, data.train_labels, 1, random=True, return_epoch=True)

    results = []
    n_iter = 0
    last_epoch = 0

    while True:
        n_iter += 1
        epoch, (X, y) = next(batch_generator)

        if epoch >= max_epochs:
            break

        X, y = X[0], y[0] # since is just 1
        
        train(model, X, y, lr_schedule, loss, n_iter)
        train_error = test(model, data.train, data.train_labels, loss)
        test_error = test(model, data.test, data.test_labels, loss)
        
        # on each epoch print the errors
        if epoch > last_epoch:
            last_epoch = epoch
            log(epoch, n_iter, train_error, test_error)
        
            results.append({
                'train_error': train_error,
                'test_error': test_error,
                'epoch': epoch,
                'hidden_size': hidden_size,
                'n_hidden': n_hidden + 1
            })

    return results


hidden_size = [5, 10, 25, 50, 100]

results = []
for width in hidden_size:
    t0 = time()
    results += run(width)
    t1 = time()
    print(f"Elapsed time for width {width}: {t1 - t0}")

df = pd.DataFrame(results)
file_name = f"neural_networks/reports/h5e2"

df.to_csv(f"{file_name}.csv", index=False)

sns.lineplot(data=df, x='epoch', y='test_error', hue='hidden_size')
plt.title("Test error by epoch")
plt.xlabel("Epoch")
plt.ylabel("Test error")

plt.savefig(f"{file_name}.png")
print("Done")




