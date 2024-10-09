import pandas as pd

class Dataset:
    def __init__(self, train, test, train_labels, test_labels):
        self.train = train
        self.test = test
        self.train_labels = train_labels
        self.test_labels = test_labels

def cars_dataset():
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    train = pd.read_csv('data/car/train.csv')
    test = pd.read_csv('data/car/test.csv')
    train.columns = cols
    test.columns = cols
    train_labels = train['class']
    train = train.drop('class', axis=1)
    test_labels = test['class']
    test = test.drop('class', axis=1)
    return Dataset(train, test, train_labels, test_labels)

def bank_dataset():
    cols = columns = [
        "age", 
        "job", 
        "marital", 
        "education", 
        "default", 
        "balance", 
        "housing", 
        "loan", 
        "contact", 
        "day", 
        "month", 
        "duration", 
        "campaign", 
        "pdays", 
        "previous", 
        "poutcome",
        "class"
    ]

    train = pd.read_csv('data/bank/train.csv')
    test = pd.read_csv('data/bank/test.csv')
    train.columns = cols
    test.columns = cols
    train_labels = train['class']
    train = train.drop('class', axis=1)
    test_labels = test['class']
    test = test.drop('class', axis=1)
    return Dataset(train, test, train_labels, test_labels)

def toy_dataset():
    cols = ['O', 'T', 'H', 'W', 'class']
    data = [
        ['S','H','H','W', '-'],
        ['S','H','H','S', '-'],
        ['O','H','H','W', '+'],
        ['R','M','H','W', '+'],
        ['R','C','N','W', '+'],
        ['R','C','N','S', '-'],
        ['O','C','N','S', '+'],
        ['S','M','H','W', '-'],
        ['S','C','N','W', '+'],
        ['R','M','N','W', '+'],
        ['S','M','N','S', '+'],
        ['O','M','H','S', '+'],
        ['O','H','N','W', '+'],
        ['R','M','H','S', '-']
    ]

    df = pd.DataFrame(data, columns=cols)
    train = df.drop('class', axis=1)
    train_labels = df['class']
    return Dataset(train, None, train_labels, None)