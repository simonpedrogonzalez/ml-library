import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from linear_regression.gradient_descent import StochasticGradientDescent, BatchGradientDescent
from linear_regression.linear_regressor import AnalyticalRegressor
from data.datasets import concrete_slump_dataset, regression_toy_dataset
from utils.stats import mse
import pandas as pd
import numpy as np
from time import time

def choose_lr(model, data):
    lrs = [1 / 2 ** i for i in range(30)]
    train, train_labels = data.train, data.train_labels
    min_iter = np.inf
    min_iter_lr = None
    for lr in lrs:
        model.lr = lr
        try:
            model.fit(train, train_labels)
        except:
            # ignore if diverges
            continue
        if model.n_iter < min_iter:
            min_iter = model.n_iter
            min_iter_lr = lr
    return min_iter_lr, min_iter

def get_best_lr(model, data):
    """Get recorded best lr or compute"""
    try:
        df = pd.read_csv('linear_regression/reports/h2e4_best_lr.csv')
        df1 = df[df['model'] == model.__class__.__name__]
        if df1.empty:
            best_lr, max_iter = choose_lr(model, data)
            df2 = pd.DataFrame([{'model': model.__class__.__name__, 'best_lr': best_lr, 'max_iter': max_iter}])
            df = pd.concat([df, df2])
            df.to_csv('linear_regression/reports/h2e4_best_lr.csv', index=False)
            return best_lr, max_iter
        else: 
            return df1['best_lr'].values[0], df1['max_iter'].values[0]
    except:
        best_lr, max_iter = choose_lr(model, data)
        df = pd.DataFrame([{'model': model.__class__.__name__, 'best_lr': best_lr, 'max_iter': max_iter}])
        df.to_csv('linear_regression/reports/h2e4_best_lr.csv', index=False)
        return best_lr, max_iter

def train_test_run(model, data):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    if model.n_iter == 0:
        # first time needs the initialization done in fit
        model.fit(train, train_labels)
        model.max_iter = None # deactivate stopping criterion to continue manually
    else:
        # continue step by step
        model.step()
    train_pred = model.predict(train)
    test_pred = model.predict(test)
    train_error = mse(train_pred, train_labels)
    test_error = mse(test_pred, test_labels)
    return train_error, test_error

def report(data):
    
    # Get the best learning rate for both models
    # First, Batch gradient descent (with bootstrap sample by default)
    # Set max iter to a big number so it isn't stopped before converging (or diverging if lr is too big)
    model = BatchGradientDescent(max_iter=10000)
    # get_best_lr reads the best lr from a file or computes it if the file doesn't exist yet
    lr, _ = get_best_lr(model, data)
    # Set the best lr to the model
    # Max iter at 1 so it stops at the first iteration, then we will continue step by step to record the errors
    model = BatchGradientDescent(lr=lr, max_iter=1)

    # The same as before but for Stochastic Gradient Descent
    model2 = StochasticGradientDescent(max_iter=10000)
    lr2, _ = get_best_lr(model2, data)
    model2 = StochasticGradientDescent(lr=lr2, max_iter=1)

    results = []
    i = 0

    while True:
        if not model.should_stop():
            train_error, test_error = train_test_run(model, data)
            results.append(['Batch Gradient Descent', model.n_iter, train_error, test_error])
        if not model2.should_stop():
            train_error2, test_error2 = train_test_run(model2, data)
            results.append(['Stochastic Gradient Descent', model2.n_iter, train_error2, test_error2])
        i += 1
        print(f"n: {i}, BGD train_e: {round(train_error, 3)}, test_e: {round(test_error, 3)}, SGD train_e: {round(train_error2, 3)}, test_e: {round(test_error2, 3)}")
        if model.should_stop() and model2.should_stop():
            break

    bgd_res = {
        'model': 'Batch Gradient Descent',
        'lr': model.lr,
        'n_iter': model.n_iter,
        'train_error': train_error,
        'test_error': test_error,
        'weights': model.w
    }

    sgd_res = {
        'model': 'Stochastic Gradient Descent',
        'lr': model2.lr,
        'n_iter': model2.n_iter,
        'train_error': train_error2,
        'test_error': test_error2,
        'weights': model2.w
    }

    weights1 = model.w
    weights2 = model2.w

    # Calculate the analytical solution
    ar = AnalyticalRegressor()
    weights3 = ar.fit(data.train, data.train_labels).w
    a_train_pred = ar.predict(data.train)
    a_test_pred = ar.predict(data.test)
    a_train_error = mse(a_train_pred, data.train_labels)
    a_test_error = mse(a_test_pred, data.test_labels)

    a_res = {
        'model': 'Analytical Regressor',
        'lr': None,
        'n_iter': None,
        'train_error': a_train_error,
        'test_error': a_test_error,
        'weights': weights3
    }


    print("Exporting report for exercise 4...")
    df = pd.DataFrame(results, columns=['model', 'n_iter', 'train_error', 'test_error'])
    df.to_csv('linear_regression/reports/h2e4_report.csv', index=False)
    
    d = [{'model': 'Batch Gradient Descent', 'weights': weights1, 'lr': model.lr},
         {'model': 'Stochastic Gradient Descent', 'weights': weights2, 'lr': model2.lr},
         {'model': 'Analytical Regressor', 'weights': weights3, 'lr': None}]
    df = pd.DataFrame(d)
    df.to_csv('linear_regression/reports/h2e4_weights.csv', index=False)

    # Create a final report with all the methods,
    # method | n iterations | learning rate | weights(round 5 dec) | Train MSE (round 5 dec) | Test MSE (round 5 dec)
    # Analytical Regressor doesn't have a learning rate or n iterations

    df = pd.DataFrame([bgd_res, sgd_res, a_res])
    df['weights'] = df['weights'].apply(lambda x: np.round(x, 5))
    df['train_error'] = df['train_error'].apply(lambda x: round(x, 5))
    df['test_error'] = df['test_error'].apply(lambda x: round(x, 5))
    df.columns = ['Method', 'Learning Rate', 'N Iterations', 'Weights', 'Train MSE', 'Test MSE']
    df.to_latex('linear_regression/reports/h2e4_final_report.tex', index=False, longtable=True)


data = concrete_slump_dataset().to_numpy()
report(data)
    




