# Telco Customer Churn
# dataset available at: https://www.kaggle.com/blastchar/telco-customer-churn/downloads/telco-customer-churn.zip/1

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing

path = 'data/customer churn/customer churn.csv'
real_data = pd.read_csv(path)

# drop the customer ID as it's not useful for this predicting churn
real_data = real_data.drop('customerID', axis=1)

# missing values are only present in TotalCharges columns, these 'missing values' only occur for customers with zero
# tenure -> may be missing because the customer did not pay anything and therefore replaced with zero
# print('Missing values:\n', (real_data == ' ').sum())  # 11 missing values for TotalCharges
real_data['TotalCharges'] = real_data['TotalCharges'].replace(" ", 0).astype('float32')


# scale the numerical values
scaler = RobustScaler()

real_data['tenure'] = scaler.fit_transform(real_data['tenure'].values.reshape(-1, 1))
real_data['MonthlyCharges'] = scaler.fit_transform(real_data['MonthlyCharges'].values.reshape(-1, 1))
real_data['TotalCharges'] = scaler.fit_transform(real_data['TotalCharges'].values.reshape(-1, 1))

# replace Yes/No columns with 1/0 columns
columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for i in range(0, len(columns)):
    real_data[columns[i]] = real_data[columns[i]].map({'Yes':1, 'No':0})

# one-hot-encode columns with multiple categorical variables
real_data = pd.get_dummies(real_data)
print(real_data)


real_data.to_pickle('data/customer churn/customer churn modified.pkl')
real_data.to_csv('data/customer churn/customer churn modified.csv')