from id3 import ID3
import pandas as pd
from utils import avg_error

def read_data():
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    train = pd.read_csv('decision-tree/data/car/train.csv')
    test = pd.read_csv('decision-tree/data/car/test.csv')
    train.columns = cols
    test.columns = cols
    train_labels = train['class']
    train = train.drop('class', axis=1)
    test_labels = test['class']
    test = test.drop('class', axis=1)
    return train, train_labels, test, test_labels


train, train_labels, test, test_labels = read_data()
metrics = ['infogain', 'majerr', 'gini']

def train_test_run(train, test, metric, max_depth):
    id3 = ID3(metric, max_depth).fit(train, train_labels)
    train_pred = id3.predict(train)
    test_pred = id3.predict(test)
    train_error = avg_error(train_pred, train_labels)
    test_error = avg_error(test_pred, test_labels)
    return train_error, test_error

def report():
    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 7)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            train_error, test_error = train_test_run(train, test, metric, max_depth)
            rows.append([metric, max_depth, train_error, test_error])
            print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}")            
    df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    print("Exporting report for exercise 2...")
    df.to_csv('decision-tree/reports/h1e2_report.csv', index=False)
    df.to_latex('decision-tree/reports/h1e2_report.tex', index=False, longtable=True)

report()