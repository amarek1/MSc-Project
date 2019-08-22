# https://www.openml.org/d/4134

import pandas as pd
from sklearn.preprocessing import RobustScaler

# data = pd.read_csv('data/bioresponse/bioresponse.csv')
# data.to_pickle('data/bioresponse/bioresponse.pkl')
data = pd.read_pickle('data/bioresponse/bioresponse.pkl')

data = data.rename(columns={'target':'class'})
data = data.replace({0:1, 1:0})

class1 = data.loc[data['class'] == 1][:350]
class0 = data.loc[data['class'] == 0]
data = pd.concat([class0,class1])

print(data)

data.to_pickle('data/bioresponse/bio_clean.pkl')