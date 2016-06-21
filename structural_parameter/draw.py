import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv', sep=' ')
# print(data)
sample = data[data.Z1 < 0.05]
# sns.distplot(sample.leq1, kde=False, hist=True, bins=np.linspace(0, 1, 21), color='r', hist_kws={"histtype": "step","linewidth": 2})
# sns.distplot(sample.leq2, kde=False, hist=True, bins=np.linspace(0, 1, 21), color='b', hist_kws={"histtype": "step","linewidth": 2})
# plt.show()
print(sample[sample.adp1 < 0.1])
