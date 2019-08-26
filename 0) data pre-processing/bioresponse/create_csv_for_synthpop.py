import pandas as pd
import pickle

data = pd.read_pickle('data/bioresponse/bio_clean.pkl')

data0 = data.loc[data['class']==0]
data1 = data.loc[data['class']==1]

data0_0 = data0.iloc[:,0:600]
data0_1 = data0.iloc[:,600:1200]
data0_2 = data0.iloc[:,1200:len(data.columns)]


data1_0 = data1.iloc[:,0:600]
data1_1 = data1.iloc[:,600:1200]
data1_2 = data1.iloc[:,1200:len(data.columns)]

data0_0.to_csv('data/bioresponse/bio_0_part1.csv')
data0_1.to_csv('data/bioresponse/bio_0_part2.csv')
data0_2.to_csv('data/bioresponse/bio_0_part3.csv')

data1_0.to_csv('data/bioresponse/bio_1_part1.csv')
data1_1.to_csv('data/bioresponse/bio_1_part2.csv')
data1_2.to_csv('data/bioresponse/bio_1_part3.csv')