import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



# load the original data
file_name = 'C:/Users/amarek/PycharmProjects/data_lab/datasets/ori_data.pkl'
ori_data = pd.read_pickle(file_name)

ori_data = ori_data.loc[ori_data['class'] == 1]
train = ori_data

label_cols = [ i for i in train.columns if 'class' in i ]
data_cols = [ i for i in train.columns if i not in label_cols ]
train_no_label = train[ data_cols ]

# projections = [ TSNE(random_state=i).fit_transform(train_no_label) for i in range(3) ]

algorithms = [['KMeans', cluster.KMeans, (), {'n_clusters': 2, 'random_state': 0}]]

rows = len(algorithms)
columns = 4
plt.figure(figsize=(columns * 3, rows * 3))
#
# for i, [name, algorithm, args, kwds] in enumerate(algorithms):
#     print(i, name)
#
#     labels = algorithm(*args, **kwds).fit_predict(train_no_label)
#
#     colors = np.clip(labels, -1, 9)
#     colors = ['C' + str(i) if i > -1 else 'black' for i in colors]
#
#     plt.subplot(rows, columns, i * columns + 1)
#     plt.scatter(train_no_label[data_cols[0]], train_no_label[data_cols[1]], c=colors)
#     plt.xlabel(data_cols[0]), plt.ylabel(data_cols[1])
#     plt.title(name)
#
#     for j in range(3):
#         plt.subplot(rows, columns, i * columns + 1 + j + 1)
#         plt.scatter(*(projections[j].T), c=colors)
#         plt.xlabel('x'), plt.ylabel('y')
#         plt.title('TSNE projection ' + str(j + 1), size=12)
#
#
# plt.suptitle('Comparison of Fraud Clusters', size=16)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

algorithm = cluster.KMeans
args, kwds = (), {'n_clusters':2, 'random_state':0}
labels = algorithm(*args, **kwds).fit_predict(train_no_label)

# print( pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) ) )

fraud_w_classes = train.copy()
fraud_w_classes['class'] = labels




