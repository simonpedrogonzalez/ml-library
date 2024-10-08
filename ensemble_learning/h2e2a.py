import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from data.datasets import bank_dataset
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_bin_median
from utils.stats import avg_error
from decision_tree.id3 import ID3
from ensemble_learning.adaboost import AdaBoost

def report(data):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    weak_learner = ID3('infogain', 1)
    adaboost = AdaBoost(weak_learner, 5).fit(train, train_labels)
    train_pred = adaboost.predict(train)
    test_pred = adaboost.predict(test)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)



    # metrics = ['infogain', 'majerr', 'gini']
    # max_depths = range(1, 7)
    # rows = []
    # total = len(metrics) * len(max_depths)
    # for i, metric in enumerate(metrics):
    #     for j, max_depth in enumerate(max_depths):
    #         train_error, test_error = train_test_run(data, metric, max_depth)
    #         rows.append([metric, max_depth, train_error, test_error])
    #         print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}")            
    # df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    # print("Exporting report for exercise 2...")
    # df.to_csv('decision_tree/reports/h1e2_report.csv', index=False)
    # df.to_latex('decision_tree/reports/h1e2_report.tex', index=False, longtable=True)


data = bank_dataset()
data.train = transform_num_to_bin_median(data.train)
data.test = transform_num_to_bin_median(data.test)
# data = dataset_to_cat_encoded_dataset(data)

report(data)