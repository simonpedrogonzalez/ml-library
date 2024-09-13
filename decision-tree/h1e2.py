from id3 import ID3
import pandas as pd

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

def avg_error(prediction, labels):
    return (prediction != labels).mean()

def train_test_run(train, test, metric, max_depth):
    id3 = ID3(metric, max_depth).fit(train, train_labels)
    print(id3.tree)
    train_pred = id3.predict(train)
    test_pred = id3.predict(test)
    train_error = avg_error(train_pred, train_labels)
    test_error = avg_error(test_pred, test_labels)
    return train_error, test_error

def report(metric, max_depth, train_error, test_error):
    print(f"Metric: {metric}, Max Depth: {max_depth}")
    print(f"Train Error: {train_error}, Test Error: {test_error}")

for metric in metrics:
    for max_depth in range(1, 7):
        train_error, test_error = train_test_run(train, test, metric, max_depth)
        report(metric, max_depth, train_error, test_error)

