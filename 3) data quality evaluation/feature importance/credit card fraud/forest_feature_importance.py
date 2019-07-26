import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats
np.random.seed(7)


def unpack_model(model_name):
    path = 'C:/Users/amarek/PycharmProjects/data_lab/models/decision tree/' + model_name
    with open(path, 'rb') as file:
        model = pickle.load(file)
        return model


unbal_model_ori = unpack_model('forest_model_unbal_ori.pkl')
bal_model_ori = unpack_model('forest_model_bal_ori.pkl')
unbal_model_mix = unpack_model('forest_model_unbal_mix.pkl')
bal_model_mix = unpack_model('forest_model_bal_mix.pkl')
unbal_model_so = unpack_model('forest_model_unbal_so.pkl')
bal_model_so = unpack_model('forest_model_bal_so.pkl')
bal_model_mix_vGAN = unpack_model('forest_model_mix_vGAN6000.pkl')
bal_model_mix_WGAN = unpack_model('forest_model_mix_WGAN4500.pkl')


# load the original data
file_name = 'C:/Users/amarek/PycharmProjects/data_lab/datasets/creditcard_normalised.csv'
ori_data = pd.read_csv(file_name)

# unbalanced data
X = ori_data.drop('class', axis=1)
y = ori_data['class']

X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = \
    train_test_split(X, y, test_size=0.2, random_state=1)


def get_feature_importance(model, x_train):
    feature_importance = pd.DataFrame(model.feature_importances_, index=x_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)
    return feature_importance


FI_unbal_ori = get_feature_importance(unbal_model_ori, X_train_unbalanced)
FI_bal_ori = get_feature_importance(bal_model_ori, X_train_unbalanced)
# FI_unbal_mix = get_feature_importance(unbal_model_mix, X_train_unbalanced)
# FI_bal_mix = get_feature_importance(bal_model_mix, X_train_unbalanced)
FI_unbal_so = get_feature_importance(unbal_model_so, X_train_unbalanced)
FI_bal_so = get_feature_importance(bal_model_so, X_train_unbalanced)
FI_bal_mix_vGAN = get_feature_importance(bal_model_mix_vGAN, X_train_unbalanced)
FI_bal_mix_WGAN = get_feature_importance(bal_model_mix_WGAN, X_train_unbalanced)

row_names_unbal_ori = list(FI_unbal_ori.index)
row_names_bal_ori = list(FI_bal_ori.index)
# row_names_unbal_mix = list(FI_unbal_mix.index)
# row_names_bal_mix = list(FI_bal_mix.index)
row_names_unbal_so = list(FI_unbal_so.index)
row_names_bal_so = list(FI_bal_so.index)
row_names_bal_mix_vGAN = list(FI_bal_mix_vGAN.index)
row_names_bal_mix_WGAN = list(FI_bal_mix_WGAN.index)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x=row_names_unbal_ori, y=FI_unbal_ori.loc[:, 'importance'], s=25, c='b', marker='o', label='unbal_ori')
ax1.scatter(x=row_names_bal_ori, y=FI_bal_ori.loc[:, 'importance'], s=25, c='r', marker='o', label='bal_ori')
# ax1.scatter(x=row_names_unbal_mix, y=FI_unbal_mix.loc[:, 'importance'], s=25, c='y', marker='o',
#             label='unbal_mix')
# ax1.scatter(x=row_names_bal_mix, y=FI_bal_mix.loc[:, 'importance'], s=25, c='g', marker='o',
#             label='bal_mix')
ax1.scatter(x=row_names_unbal_so, y=FI_unbal_so.loc[:, 'importance'], s=30, c='b', marker='x',
            label='unbal_so')
ax1.scatter(x=row_names_bal_so, y=FI_bal_so.loc[:, 'importance'], s=30, c='r', marker='x',
            label='bal_so')



plt.xlabel('features')
plt.ylabel('importance score (x$10^-2$)')
plt.legend(loc='upper right')
plt.xticks(fontsize=10, rotation=90)
plt.title('Random forest feature importance')
plt.grid()
plt.show()


# assess the similarity of feature importance distribution


merged_FI_unbal = pd.concat([FI_unbal_ori, FI_unbal_so], axis=1, sort=True)
merged_FI_bal = pd.concat([FI_bal_ori, FI_bal_so], axis=1, sort=True)


# fig = plt.figure()
# plt.suptitle("How Pearson's correlation coefficient works - how far are the points from the line")
# ax1 = fig.add_subplot(211)
# ax1.scatter(merged_FI_bal.iloc[:, 0], merged_FI_bal.iloc[:, 1], s=25, c='b', marker='o', label='unbal_ori')
# plt.title('balanced')
# plt.xlabel('original feature importance score')
# plt.ylabel('synthetic feature importance score')
#
# ax2 = fig.add_subplot(212)
# ax2.scatter(merged_FI_unbal.iloc[:, 0], merged_FI_unbal.iloc[:, 1], s=25, c='b', marker='o', label='unbal_ori')
# plt.title('unbalanced')
# plt.xlabel('original feature importance score')
# plt.ylabel('synthetic feature importance score')
# plt.tight_layout()
# plt.show()


# print('the averaged distance for the syn and ori data(unbalanced):', sum(abs(FI_unbal_ori.values-FI_unbal_so.values)))
# print('the averaged distance for the syn and ori data(balanced):', sum(abs(FI_bal_ori.values-FI_bal_so.values)))
#
# # dist_unbal = float(sum(abs(FI_unbal_ori.values-FI_unbal_so.values)))
# # dist_bal = float(sum(abs(FI_bal_ori.values-FI_bal_so.values)))
#
# dist_unbal = float(sum(abs(FI_unbal_ori.values-FI_unbal_so.values)/FI_unbal_ori.values))
# dist_bal = float(sum(abs(FI_bal_ori.values-FI_bal_so.values)/FI_bal_ori.values))
# print(FI_unbal_ori.values-FI_unbal_so.values)
# print((FI_unbal_ori.values-FI_unbal_so.values)/FI_unbal_ori.values)
#
# fig = plt.figure()
# x = np.arange(2)
# distance = [dist_unbal, dist_bal]
# plt.title('Normalised distance between original and synthetic feature importance')
# plt.ylabel('distance')
# plt.bar(x, distance)
# plt.xticks(x, ('unbalanced model', 'balanced model'))
# plt.show()
