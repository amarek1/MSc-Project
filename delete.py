import pandas as pd
import pickle

file_name = 'C:/Users/amarek/Desktop/R/synthpop/syntpop_data_cart_fo_15000.csv'
a = pd.read_csv(file_name)

b = a.to_pickle('data/credit card fraud/synthpop_fo_15000.pkl')



file_name = 'data/credit card fraud/synthpop_fo_15000.pkl'
ori_data = pd.read_pickle(file_name)
print(ori_data)
