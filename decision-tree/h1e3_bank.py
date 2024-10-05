from id3 import ID3
import pandas as pd
from utils import avg_error, CatEncodedDataFrame, CatEncodedSeries
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

    return train, train_labels, test, test_labels

def preprocess_data(train, train_labels, test, test_labels):
    # Binarize numerical features
    # This is not done in id3 code because
    # it should be a preprocessing step, not part
    # of the ID3 algorithm.
    # Note: in this dataset, numerical cols do not have missing ("unknown") values.
    train = transform_num_to_bin_median(train)
    test = transform_num_to_bin_median(test)

    # Create copies with impute values
    # Impute unknown values with the most common value
    # for each column. This is done after binarizing the
    # numerical features.
    train_imp = train.copy()
    test_imp = test.copy()
    train_imp = impute_mode(train_imp, "unknown")
    test_imp = impute_mode(test_imp, "unknown")

    train_imp = CatEncodedDataFrame().from_pandas(train_imp)
    test_imp = CatEncodedDataFrame().from_pandas(test_imp)

    train = CatEncodedDataFrame().from_pandas(train)
    train_labels = CatEncodedSeries().from_pandas(train_labels)
    test = CatEncodedDataFrame().from_pandas(test)
    test_labels = CatEncodedSeries().from_pandas(test_labels)

    with_unknown = (train, train_labels, test, test_labels)
    with_imp_values = (train_imp, train_labels, test_imp, test_labels)

    return with_unknown, with_imp_values

with_unknown, with_imp_values = preprocess_data(*read_data())
metrics = ['infogain', 'majerr', 'gini']

def train_test_run(train, test, metric, max_depth, train_labels, test_labels):
    id3 = ID3(metric, max_depth).fit(train, train_labels)
    train_pred = id3.predict(train)
    test_pred = id3.predict(test)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)
    return train_error, test_error

def report3b():
    """ Generate trees with depths 1 to 16 using the three metrics
    for the bank marketing dataset. Export the results of the train
    and test errors to a CSV file and a LaTeX table.
    """
    train, train_labels, test, test_labels = with_unknown
    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 17)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            t0 = time()
            train_error, test_error = train_test_run(train, test, metric, max_depth, train_labels, test_labels)
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
    train_imp, train_labels, test_imp, test_labels = with_imp_values

    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 17)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            t0 = time()
            train_error, test_error = train_test_run(train_imp, test_imp, metric, max_depth, train_labels, test_labels)
            t1 = time()
            print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}, Time: {t1 - t0:.2f}s")            
            rows.append([metric, max_depth, train_error, test_error])
    print("Exporting report for exercise 3c...")
    df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    df.to_csv('decision-tree/reports/h1e3c_report.csv', index=False)
    df.to_latex('decision-tree/reports/h1e3c_report.tex', index=False, longtable=True)


report3b()
report3c()
# train, train_labels, test, test_labels = with_unknown
# id3 = ID3('infogain', 3).fit(train, train_labels)
# print(id3.tree)
