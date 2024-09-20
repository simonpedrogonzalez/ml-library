from id3 import ID3
import pandas as pd
from utils import avg_error
from time import time
from preprocessing import transform_num_to_bin_median, impute_mode

def read_data():
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

    train = pd.read_csv('decision-tree/data/bank/train.csv')
    test = pd.read_csv('decision-tree/data/bank/test.csv')
    train.columns = cols
    test.columns = cols
    train_labels = train['class']
    train = train.drop('class', axis=1)
    test_labels = test['class']
    test = test.drop('class', axis=1)

    # Binarize numerical features
    # This is not done in id3 code because
    # it should be a preprocessing step, not part
    # of the ID3 algorithm.
    # Note: in this dataset, numerical cols do not have missing ("unknown") values.
    train = transform_num_to_bin_median(train)
    test = transform_num_to_bin_median(test)

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

def report3b():
    """ Generate trees with depths 1 to 16 using the three metrics
    for the bank marketing dataset. Export the results of the train
    and test errors to a CSV file and a LaTeX table.
    """
    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 17)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            t0 = time()
            train_error, test_error = train_test_run(train, test, metric, max_depth)
            t1 = time()
            print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}, Time: {t1 - t0:.2f}s")            
            rows.append([metric, max_depth, train_error, test_error])
    print("Exporting report for exercise 3b...")
    df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    df.to_csv('decision-tree/reports/h1e3b_report.csv', index=False)
    df.to_latex('decision-tree/reports/h1e3b_report.tex', index=False, longtable=True)

def report3c():
    """ Run the same experiment as report3b but using
    imputed data instead of the unknown class.
    """

    # Impute unknown values with the most common value
    # for each column. This is done after binarizing the
    # numerical features.
    train_imp = impute_mode(train, "unknown")
    test_imp = impute_mode(test, "unknown")

    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 17)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            t0 = time()
            train_error, test_error = train_test_run(train_imp, test_imp, metric, max_depth)
            t1 = time()
            print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}, Time: {t1 - t0:.2f}s")            
            rows.append([metric, max_depth, train_error, test_error])
    print("Exporting report for exercise 3c...")
    df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    df.to_csv('decision-tree/reports/h1e3c_report.csv', index=False)
    df.to_latex('decision-tree/reports/h1e3c_report.tex', index=False, longtable=True)


report3b()
report3c()
