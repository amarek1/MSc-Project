import pandas as pd

file_name = '2) synthetic data generation/tGAN/customer churn/churn/tGAN_churn_50002.pkl'
syn_fraud = pd.read_pickle(file_name)



syn_fraud['class'] = syn_fraud['class'].astype('float64')

print(syn_fraud.dtypes)

syn_fraud.to_pickle(file_name)