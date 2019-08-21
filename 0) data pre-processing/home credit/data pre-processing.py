# pre-processing based on: https://www.kaggle.com/shivamb/homecreditrisk-extensive-eda-baseline-0-772/notebook

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# print('1')
# app_train = pd.read_csv('data/home credit/application_train.csv')
# app_train.to_pickle('data/home credit/application_train.pkl')
# print('2')
# bureau = pd.read_csv('data/home credit/bureau.csv')
# bureau.to_pickle('data/home credit/bureau.pkl')
# print('3')
# bureau_balance = pd.read_csv('data/home credit/bureau_balance.csv')
# bureau_balance.to_pickle('data/home credit/bureau_balance.pkl')
# print('4')
# card = pd.read_csv('data/home credit/credit_card_balance.csv')
# card.to_pickle('data/home credit/credit_card_balance.pkl')
# print('5')
# installments = pd.read_csv('data/home credit/installments_payments.csv')
# installments.to_pickle('data/home credit/installments_payments.pkl')
# print('6')
# POS = pd.read_csv('data/home credit/POS_CASH_balance.csv')
# POS.to_pickle('data/home credit/POS_CASH_balance.pkl')
# print('7')
# previous_app = pd.read_csv('data/home credit/previous_application.csv')
# previous_app.to_pickle('data/home credit/previous_application.pkl')

# # pickle opens quicker than csv
# print('1')
# app_train = pd.read_pickle('data/home credit/application_train.pkl')
# print('2')
# bureau = pd.read_pickle('data/home credit/bureau.pkl')
# print('3')
# bureau_balance = pd.read_pickle('data/home credit/bureau_balance.pkl')
# print('4')
# card = pd.read_pickle('data/home credit/credit_card_balance.pkl')
# print('5')
# installments = pd.read_pickle('data/home credit/installments_payments.pkl')
# print('6')
# POS = pd.read_pickle('data/home credit/POS_CASH_balance.pkl')
# print('7')
# previous_app = pd.read_pickle('data/home credit/previous_application.pkl')
#
#
# # get a list of categorical fatures
# def _get_categorical_features(data):
#     features = [col for col in list(data.columns) if data[col].dtype == 'object']
#     return features
#
#
# # create dummy variables
# def _get_dummies(data, features):
#     for col in features:
#         data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
#     return data
#
#
# # create dummy variables for given datasets
# app_train_cat = _get_categorical_features(app_train)
# previous_app_cat = _get_categorical_features(previous_app)
# bureau_cat = _get_categorical_features(bureau)
# POS_cat = _get_categorical_features(POS)
# card_cat = _get_categorical_features(card)
#
# app_train = _get_dummies(app_train, app_train_cat)
# previous_app = _get_dummies(previous_app, previous_app_cat)
# bureau = _get_dummies(bureau, bureau_cat)
# POS = _get_dummies(POS, POS_cat)
# card = _get_dummies(card, card_cat)
#
# # count the number of previous applications for and ID, average and attach to app_train
# prev_apps_count = previous_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
# previous_app['SK_ID_PREV'] = previous_app['SK_ID_CURR'].map(prev_apps_count['SK_ID_PREV'])
#
# prev_apps_avg = previous_app.groupby('SK_ID_CURR').mean()
# prev_apps_avg.columns = ['p_' + col for col in prev_apps_avg.columns]
# app_train = app_train.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')
#
# # average values for all bureau features and attach to app_train
# bureau_avg = bureau.groupby('SK_ID_CURR').mean()
# bureau_avg['buro_count'] = bureau[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
# bureau_avg.columns = ['b_' + f_ for f_ in bureau_avg.columns]
# app_train = app_train.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')
#
# # count the previous installments, average values and attach to app_train
# cnt_inst = installments[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
# installments['SK_ID_PREV'] = installments['SK_ID_CURR'].map(cnt_inst['SK_ID_PREV'])
#
# avg_inst = installments.groupby('SK_ID_CURR').mean()
# avg_inst.columns = ['i_' + f_ for f_ in avg_inst.columns]
# app_train = app_train.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
#
# # count and average POS and attach to app_train
# POS_count = POS[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
# POS['SK_ID_PREV'] = POS['SK_ID_CURR'].map(POS_count['SK_ID_PREV'])
#
# POS_avg = POS.groupby('SK_ID_CURR').mean()
# app_train = app_train.merge(right=POS_avg.reset_index(), how='left', on='SK_ID_CURR')
#
# # count and average credit card balance and attach to app_train
# nb_prevs = card[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
# card['SK_ID_PREV'] = card['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
#
# avg_cc_bal = card.groupby('SK_ID_CURR').mean()
# avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
# app_train = app_train.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
#




app_train = pd.read_pickle('data/home credit/application_train.pkl')

# replace nan with 0
print(len(app_train),len(app_train.columns))
app_train = app_train.dropna(thresh=119)
print(len(app_train),len(app_train.columns))
app_train = app_train.dropna(axis='columns')
print(len(app_train),len(app_train.columns))

app_train = app_train.drop(['SK_ID_CURR'], axis=1)

app_train = pd.get_dummies(app_train)

target = app_train['TARGET']
app_train = app_train.drop(['TARGET'], axis=1)
app_train['TARGET'] = target

data=app_train
# rescale columns which ar enot categorical or within -1 and 1 range
# to_rescale = []
# for col in data.columns:
#     if data[col].max() > 1 or data[col].min() < -1:
#         to_rescale.append(col)
#
# rob_scaler = RobustScaler()
# for i in to_rescale:
#     data[i] = rob_scaler.fit_transform(data[i].values.reshape(-1,1))

data = data.astype('float64')
# rename TARGET to class for consistency
data = data.rename(columns={'TARGET':'class'})

print(len(data),len(data.columns))
print('class0',len(data.loc[data['class'] == 0]),'class1',len(data.loc[data['class'] == 1]))
data['class'] = data['class'].astype('int')

data.to_pickle('data/home credit/home_clean.pkl')