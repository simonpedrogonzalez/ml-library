from id3 import id3
import pandas as pd

def read_data():
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    categories = {
        'buying': ['vhigh', 'high', 'med', 'low'],
        'maint': ['vhigh', 'high', 'med', 'low'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high'],
        'class': ['unacc', 'acc', 'good', 'vgood']
    }
    train = pd.read_csv('data/car/train.csv')
    test = pd.read_csv('data/car/test.csv')
    train.columns = cols
    test.columns = cols
    train = train.astype('category')
    test = test.astype('category')
    train = train.apply(lambda x: x.cat.codes)
    test = test.apply(lambda x: x.cat.codes)
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
    tree = ID3(metric, max_depth).fit(train, train_labels)
    train_pred = tree.predict(train)
    test_pred = tree.predict(test)
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

