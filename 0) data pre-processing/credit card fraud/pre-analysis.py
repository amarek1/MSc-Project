import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

data = pd.read_csv('C:/Users/amarek/PycharmProjects/data_lab/creditcard_normalised.csv')

# print(data.loc[1:3,'V1'])

# check for missing values
data.isnull().sum()

# transactions over time
# basic statistics
# print('Fraud')
# print(data.Time[data.Class == 1].describe())
# print('Normal')
# print(data.Time[data.Class == 0].describe())
#
# # number of transactions over time
#
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
#
# bins = 50
#
# ax1.hist(data.Time[data.Class == 1], bins = bins)
# ax1.set_title('Fraud')
#
# ax2.hist(data.Time[data.Class == 0], bins = bins)
# ax2.set_title('Normal')
#
# plt.xlabel('Time (in Seconds)')
# plt.ylabel('Number of Transactions')
#
# plt.show()
# #
#
# # amount distribution
#
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
#
# bins = 30
#
# ax1.hist(data.Amount[data.Class == 1], bins = bins)
# ax1.set_title('Fraud')
#
# ax2.hist(data.Amount[data.Class == 0], bins = bins)
# ax2.set_title('Normal')
#
# plt.xlabel('Amount ($)')
# plt.ylabel('Number of Transactions')
# plt.yscale('log')
# plt.show()

# features comparison between fraud and normal

# get column names for features
v_features = data.iloc[:,1:29]

# plt.figure(figsize=(12, 28*4))
# gs = gridspec.GridSpec(28, 1)
# for i, cn in enumerate(data[v_features]):
#     ax = plt.subplot(gs[i])
#     sns.distplot(data[cn][data.Class == 1], bins=50)
#     sns.distplot(data[cn][data.Class == 0], bins=50)
#     ax.set_xlabel('')
#     ax.set_title('histogram of feature:' + str(cn))
# plt.show()

fig = plt.figure()
for i in range(1,29):
    ax = fig.add_subplot(4,7,i)
    sns.distplot(data.iloc[:, i][data['class'] == 1], bins=50, label='fraud')
    sns.distplot(data.iloc[:, i][data['class'] == 0], bins=50, label='normal')
# plt.tight_layout()
plt.show()






