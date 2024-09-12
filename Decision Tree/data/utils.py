def entropy(y):
    proba = y.value_counts(normalize=True)
    return -np.sum(proba * np.log2(proba))

def information_gain(X, y, feature):
    total_entropy = self.entropy(y)
    feature_values = X[feature].unique()
    weights = X[feature].value_counts(normalize=True)
    subset_entropies = []
    for value in feature_values:
        subset = y[X[feature] == value]
        subset_entropy = self.entropy(subset)
        subset_entropies.append(subset_entropy)
    return total_entropy - np.sum(weights * np.array(subset_entropies))

def majority_error(y):
    return 1 - y.value_counts(normalize=True).max()

def gini_index(y):
    proba = y.value_counts(normalize=True)
    return 1 - np.sum(proba ** 2)

def min_argmin(arr):
    return min(enum(arr), key=lambda x: x[1])

def max_argmax(arr):
    return max(enum(arr), key=lambda x: x[1])