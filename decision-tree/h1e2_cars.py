from id3 import ID3
import pandas as pd
from utils import avg_error, CatEncodedSeries, CatEncodedDataFrame

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

def preprocess_data(train, train_labels, test, test_labels):
    train = CatEncodedDataFrame().from_pandas(train)
    train_labels = CatEncodedSeries().from_pandas(train_labels)
    test = CatEncodedDataFrame().from_pandas(test)
    test_labels = CatEncodedSeries().from_pandas(test_labels)
    return train, train_labels, test, test_labels

# train2, train_labels2, test2, test_labels2 = preprocess_data(*read_data())
train, train_labels, test, test_labels = preprocess_data(*read_data())
metrics = ['infogain', 'majerr', 'gini']

# @profile
def train_test_run(train, test, metric, max_depth):
    id3 = ID3(metric, max_depth).fit(train, train_labels)
    train_pred = id3.predict(train)
    test_pred = id3.predict(test)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)

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