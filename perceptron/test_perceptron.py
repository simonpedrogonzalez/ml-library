import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from perceptron.perceptron import Perceptron, VotedPerceptron
from data.datasets import linearly_separable_toy_dataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = linearly_separable_toy_dataset()
X = data.train
y = data.train_labels

p = VotedPerceptron(max_epochs=1000, lr=0.1)
p.fit(X, y)
pred = p.predict(X)
print(f"train error: {np.mean(pred != y)}")

# draw points, labeled by color
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
# draw decision boundary
w = p.w
linspace = np.linspace(X.min(), X.max())
plt.plot(linspace, (-w[0] - w[1] * linspace) / w[2])
plt.show()

print('done')