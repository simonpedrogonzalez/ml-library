import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from data.datasets import bank_note_dataset
from utils.stats import cost
import pandas as pd
import numpy as np
from time import time
from svm.svm import DualSVM, KernelSVM, Kernel
from itertools import product
from utils.stats import avg_error

data = bank_note_dataset().to_numpy()

def train_test_run(model, data):
    
    t0 = time()
    model.fit(data.train, data.train_labels)
    
    train_pred = model.predict(data.train)
    test_pred = model.predict(data.test)
    t1 = time()

    train_error = avg_error(train_pred, data.train_labels)
    test_error = avg_error(test_pred, data.test_labels)

    result = {
        'model': model.__class__.__name__,
        'C': model.C,
        'kernel_gamma': model.kernel.params['gamma'] if isinstance(model, KernelSVM) else None,
        'test_error': test_error,
        'train_error': train_error,
        'weights': model.w
    }

    print(f"model: {model.__class__.__name__}, C: {model.C}, kernel_gamma: {model.kernel.params['gamma'] if isinstance(model, KernelSVM) else None}, train_error: {train_error}, test_error: {test_error}, time: {round(t1 - t0)}")
    return result

def report():
    np.random.seed(42)

    def gaussian_kernel(X, _X, gamma):
        X_norm = np.sum(X**2, axis=1)[:, None]  # Shape (n_samples_X, 1)
        _X_norm = np.sum(_X**2, axis=1)[None, :]  # Shape (1, n_samples__X)
        dist_squared = X_norm + _X_norm - 2 * X @ _X.T
        return np.exp(-dist_squared / gamma)

    kernel_gammas = [0.1, 0.5, 1.5, 1, 5, 100]
    Cs = [100. / 873., 500. / 873., 700. / 873.]

    models = []
    for C in Cs:
        models.append(DualSVM(C=C))
        for gamma in kernel_gammas:
            kernel = Kernel(gaussian_kernel, {'gamma': gamma})
            models.append(KernelSVM(C=C, kernel=kernel))

    results = []

    for model in models:
        results.append(train_test_run(model, data))
        

    df = pd.DataFrame(results)
    

    df_copy = df.copy()
    df_copy.drop(columns=['weights'], inplace=True)
    df_copy.to_csv('svm/reports/h4e3_results.csv', index=False)
    df_copy.to_latex('svm/reports/h4e3_results.tex', index=False)

    weights_expanded = pd.DataFrame(df['weights'].tolist(), columns=[f'w{i}' for i in range(len(df['weights'].iloc[0]))])
    df_final = pd.concat([df.drop(columns=['weights']), weights_expanded], axis=1)

    non_zero_counts = [np.sum(np.abs(model.alphas) > 1e-6) for model in models]
    df_final['n_support_vectors'] = non_zero_counts
    
    for col in df_final.columns:
        if df_final[col].dtype == 'float64':
            df_final[col] = df_final[col].apply(lambda x: int(x) if x.is_integer() else round(x, 3))

    df_final.to_csv('svm/reports/h4e3_final.csv', index=False)
    df_final = df_final.astype(str)
    df_final.to_latex('svm/reports/h4e3_final.tex', index=False, na_rep='', escape=False) 

report()