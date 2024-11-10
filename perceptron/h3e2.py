import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from perceptron.perceptron import Perceptron, VotedPerceptron, AveragedPerceptron
from data.datasets import bank_note_dataset
from utils.stats import avg_error
import numpy as np
import pandas as pd

def train_test_run(model, data):
    model.fit(data.train, data.train_labels)

    train_pred = model.predict(data.train)
    test_pred = model.predict(data.test)

    train_error = avg_error(train_pred, data.train_labels)
    test_error = avg_error(test_pred, data.test_labels)

    result = {
        'model': model.__class__.__name__,
        'train_error': train_error,
        'test_error': test_error,
        'epochs': model.epoch,
        'learning_rate': model.lr,
        'weights': model.a if hasattr(model, 'a') else model.w,
        'c': model.c if hasattr(model, 'c') else None
    }

    print(f"model: {model.__class__.__name__}, train_error: {train_error}, test_error: {test_error}")
    return result

def report():
    np.random.seed(42)
    results = []
    lr = 1
    epochs = 10
    data = bank_note_dataset().to_numpy()
    model1 = Perceptron(max_epochs=epochs, lr=lr)
    model2 = VotedPerceptron(max_epochs=epochs, lr=lr)
    model3 = AveragedPerceptron(max_epochs=epochs, lr=lr)
    models = [model1, model2, model3]
    for model in models:
        results.append(train_test_run(model, data))
    df = pd.DataFrame(results)
    
    df.drop(columns=['weights'], inplace=True)
    df.drop(columns=['c'], inplace=True)

    df.to_csv('perceptron/reports/h3_results.csv', index=False)
    df.to_latex('perceptron/reports/h3_results.tex', index=False)

    df_w = pd.DataFrame()
    for res in results:
        model_name = res['model']
        w = res['weights']
        c = res['c']
        if model_name == 'VotedPerceptron':
            dfvp = pd.DataFrame(w, columns=[f'w{i}' for i in range(w.shape[1])])
            dfvp['model'] = model_name
            dfvp['c'] = c
            df_w = pd.concat([df_w, dfvp])
        else:
            dfp = pd.DataFrame(w.reshape(1, -1), columns=[f'w{i}' for i in range(w.shape[0])])
            dfp['model'] = model_name
            dfp['c'] = c
            df_w = pd.concat([df_w, dfp])
    


    for col in df_w.columns:
        if col not in ['model', 'c']:
            if df_w[col].dtype == 'float64':
                df_w[col] = df_w[col].apply(lambda x: int(x) if x.is_integer() else round(x, 3))




    df_w.to_csv('perceptron/reports/h3_weights.csv', index=False)

    df_w = df_w.astype(str)

    df_w.to_latex('perceptron/reports/h3_weights.tex', index=False, na_rep='', escape=False) 

report()