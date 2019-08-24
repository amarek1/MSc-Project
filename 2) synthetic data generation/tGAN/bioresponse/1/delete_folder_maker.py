import pandas as pd
import pickle

data = pd.read_pickle('data/bioresponse/bio_short.pkl')

data0 = data.loc[data['class']==0]
data1 = data.loc[data['class']==1]

data0.to_csv('data/bioresponse/bio_0.csv')
data1.to_csv('data/bioresponse/bio_1.csv')
