import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import pandas as pd
from time import time
from decision_tree.id3 import ID3
from utils.stats import avg_error
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_bin_median, impute_mode
from data.datasets import bank_dataset

def preprocess(data):
    data.train = transform_num_to_bin_median(data.train)
    data.test = transform_num_to_bin_median(data.test)
    data_with_unknown = dataset_to_cat_encoded_dataset(data)
    data.train = impute_mode(data.train, "unknown")
    data.test = impute_mode(data.test, "unknown")
    data_with_imp_values = dataset_to_cat_encoded_dataset(data)
    return data_with_unknown, data_with_imp_values

# def preprocess_data(train, train_labels, test, test_labels):
#     # Binarize numerical features
#     # This is not done in id3 code because
#     # it should be a preprocessing step, not part
#     # of the ID3 algorithm.
#     # Note: in this dataset, numerical cols do not have missing ("unknown") values.
#     train = transform_num_to_bin_median(train)
#     test = transform_num_to_bin_median(test)

#     # Create copies with impute values
#     # Impute unknown values with the most common value
#     # for each column. This is done after binarizing the
#     # numerical features.
#     train_imp = train.copy()
#     test_imp = test.copy()
#     train_imp = impute_mode(train_imp, "unknown")
#     test_imp = impute_mode(test_imp, "unknown")

#     train_imp = CatEncodedDataFrame().from_pandas(train_imp)
#     test_imp = CatEncodedDataFrame().from_pandas(test_imp)

#     train = CatEncodedDataFrame().from_pandas(train)
#     train_labels = CatEncodedSeries().from_pandas(train_labels)
#     test = CatEncodedDataFrame().from_pandas(test)
#     test_labels = CatEncodedSeries().from_pandas(test_labels)

#     with_unknown = (train, train_labels, test, test_labels)
#     with_imp_values = (train_imp, train_labels, test_imp, test_labels)

#     return with_unknown, with_imp_values

# with_unknown, with_imp_values = preprocess_data(*read_data())
# metrics = ['infogain', 'majerr', 'gini']

def train_test_run(data, metric, max_depth):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    id3 = ID3(metric, max_depth).fit(train, train_labels)
    train_pred = id3.predict(train)
    test_pred = id3.predict(test)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)
    return train_error, test_error

# def report3b():
#     """ Generate trees with depths 1 to 16 using the three metrics
#     for the bank marketing dataset. Export the results of the train
#     and test errors to a CSV file and a LaTeX table.
#     """
#     train, train_labels, test, test_labels = with_unknown
#     metrics = ['infogain', 'majerr', 'gini']
#     max_depths = range(1, 17)
#     rows = []
#     total = len(metrics) * len(max_depths)
#     for i, metric in enumerate(metrics):
#         for j, max_depth in enumerate(max_depths):
#             t0 = time()
#             train_error, test_error = train_test_run(train, test, metric, max_depth, train_labels, test_labels)
#             t1 = time()
#             print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}, Time: {t1 - t0:.2f}s")            
#             rows.append([metric, max_depth, train_error, test_error])
#     print("Exporting report for exercise 3b...")
#     df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
#     df.to_csv('decision-tree/reports/h1e3b_report.csv', index=False)
#     df.to_latex('decision-tree/reports/h1e3b_report.tex', index=False, longtable=True)

def report(data, exercise):

    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 17)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            t0 = time()
            train_error, test_error = train_test_run(data, metric, max_depth)
            t1 = time()
            print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}, Time: {t1 - t0:.2f}s")            
            rows.append([metric, max_depth, train_error, test_error])
    print("Exporting report for exercise 3c...")
    df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    df.to_csv(f"decision_tree/reports/h1e{exercise}_report.csv", index=False)
    df.to_latex(f"decision_tree/reports/h1e{exercise}_report.tex", index=False, longtable=True)

data = bank_dataset()
data_with_unknown, data_with_imp_values = preprocess(data)
report(data_with_unknown, '3b')
report(data_with_imp_values, '3c')
