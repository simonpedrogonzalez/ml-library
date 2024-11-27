import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from data.datasets import bank_note_dataset
from utils.stats import cost
import pandas as pd
import numpy as np
from time import time
from svm.svm import PrimalSGDSVM, LearningRateSchedule
from itertools import product
from utils.stats import avg_error

data = bank_note_dataset().to_numpy()

def calculate_best_lr_parameters(model):
    params = model.lr_schedule.params
    for k in params.keys():
        params[k] = np.linspace(0.01, 0.1, 10).tolist() + np.linspace(0.1, 1, 10).tolist()
    best = None
    min_n_iter = np.inf
    for combination in product(*params.values()):
        current_params = dict(zip(params.keys(), combination))
        model.lr_schedule.params = current_params
        try:
            model.fit(data.train, data.train_labels)
            if model.n_iter < min_n_iter:
                best = {
                    'model': model.__class__.__name__,
                    'C': model.C,
                    'lr_schedule': model.lr_schedule.function.__name__,
                    'lr_schedule_params': current_params,
                    'n_iter': model.n_iter,
                    'epoch': model.epoch
                }
                min_n_iter = model.n_iter
                print(f"New best: {best}")
            else:
                print(f"Worse at {current_params}")
        except:
            print("Not converged")
            pass
    filename = f"svm/reports/h4e2_schedules_params.csv"
    try:
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([best])])
        df.to_csv(filename, index=False)
    except FileNotFoundError:
        results = [best]
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
    return best['lr_schedule_params']

def search_best_lr_parameters(model, df):
    # same name, C, and lr_schedule
    df1 = df[(df['model'] == model.__class__.__name__) & (df['C'] == model.C) & (df['lr_schedule'] == model.lr_schedule.function.__name__)]
    if df1.empty:
        return False
    p = df1.iloc[0]['lr_schedule_params']
    return eval(p)

def get_best_lr_parameters(model):
    lr_schedule = model.lr_schedule
    C = model.C
    filename = f"svm/reports/h4e2_schedules_params.csv"
    df = None
    try:
        df = pd.read_csv(filename)
        best_params = search_best_lr_parameters(model, df)
        if not best_params:
            raise ValueError("Not found")
    except: # file not found or entry not in file
        best_params = calculate_best_lr_parameters(model)
    return best_params

def train_test_run(model, data):
    if model.n_iter is None:
        # first time initialization in fit
        old_max_epochs = model.max_epochs
        model.max_epochs = 0
        model.fit(data.train, data.train_labels)
        model.max_epochs = old_max_epochs
    else:
        # continue step by step
        model.step()

    train_pred = model.predict(data.train)
    test_pred = model.predict(data.test)

    train_error = avg_error(train_pred, data.train_labels)
    test_error = avg_error(test_pred, data.test_labels)

    result = {
        't': model.n_iter,
        'model': model.__class__.__name__,
        'C': model.C,
        'lr_schedule': model.lr_schedule.function.__name__,
        'test_error': test_error,
        'train_error': train_error,
        'n_iter': model.n_iter,
        'epochs': model.epoch,
        'weights': model.w
    }

    print(f"t: {model.n_iter}, C: {model.C}, lr_schedule: {model.lr_schedule.function.__name__}, train_error: {train_error}, test_error: {test_error}")
    return result

def report():
    np.random.seed(42)
    epochs = 100

    def schedule1(t, lr0, a):
        return lr0 / (1 + lr0 / a * t)
    
    def schedule2(t, lr0):
        return lr0 / (1 + t)
    
    lrs1 = LearningRateSchedule(schedule1, {'lr0': None, 'a': None}) # parameters are going to be set by the search function
    lrs2 = LearningRateSchedule(schedule2, {'lr0': None})

    Cs = [100. / 873., 500. / 873., 700. / 873.]

    models = []
    for i in range(len(Cs)):
        models.append(PrimalSGDSVM(lr_schedule=lrs1, C=Cs[i], max_epochs=100))
        models.append(PrimalSGDSVM(lr_schedule=lrs2, C=Cs[i], max_epochs=100))
    
    for model in models:
        model.lr_schedule.params = get_best_lr_parameters(model)

    results = []

    for model in models:
        while True:
            results.append(train_test_run(model, data))
            if model.should_stop():
                break
    
    df = pd.DataFrame(results)
    

    df_copy = df.copy()
    df_copy.drop(columns=['weights'], inplace=True)
    df_copy.to_csv('svm/reports/h4e2_results.csv', index=False)
    df_copy.to_latex('svm/reports/h4e2_results.tex', index=False)

    df_final = pd.DataFrame()
    for model in models:
        df1 = df[(df['model'] == model.__class__.__name__) & (df['C'] == model.C) & (df['lr_schedule'] == model.lr_schedule.function.__name__)]
        # get max n_iter
        df1 = df1[df1['n_iter'] == df1['n_iter'].max()]
        df1.drop(columns=['weights'], inplace=True)
        ws = model.w
        for i in range(ws.shape[0]):
            df1[f'w{i}'] = ws[i]
        df_final = pd.concat([df_final, df1])
    
    for col in df_final.columns:
        if col[:1] == 'w':
            if df_final[col].dtype == 'float64':
                df_final[col] = df_final[col].apply(lambda x: int(x) if x.is_integer() else round(x, 3))

    df_final.to_csv('svm/reports/h4e2_final.csv', index=False)

    df_final = df_final.astype(str)

    df_final.to_latex('svm/reports/h4e2_final.tex', index=False, na_rep='', escape=False) 

report()