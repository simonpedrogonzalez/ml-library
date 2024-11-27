import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


file = "svm/reports/h4e2_results.csv"

df = pd.read_csv(file)

print('Creating plots')
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='t', y='test_error', hue='C', style='lr_schedule', palette=sns.color_palette("hls", 3))
plt.title('Test error by step t, C and lr_schedule')
plt.savefig('svm/reports/h4e2_test_error.png')
# plt.show()

# the same but with train_error
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='t', y='train_error', hue='C', style='lr_schedule', palette=sns.color_palette("hls", 3))
plt.title('Train error by step t, C and lr_schedule')
plt.savefig('svm/reports/h4e2_train_error.png')
