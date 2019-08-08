import pandas as pd
import numpy as np

file_name = 'data/satisfaction/satisfaction.csv'
data = pd.read_csv(file_name)

stats = data.describe()
# stats.to_csv('data/satisfaction/stats.csv')  # easier to quickly view in excel

# check number of 0 and 1 class
print(data['TARGET'].value_counts())

# remove columns which are constant for all rows
remove = []
for col in data.columns:
    if data[col].std() == 0:
        remove.append(col)
data.drop(remove, axis=1, inplace=True)

# remove ID
data.drop(['ID'], axis=1)
print(len(data.columns))

# remove duplicate columns
remove_dup = []
cols = data.columns
for i in range(0,len(cols)-1):
    a = data[cols[i]].values
    for j in range(i+1, len(cols)):
        b = data[cols[j].values]
        if np.array_equal(a, b):
            remove_dup.append(cols[j])

print(len(remove_dup))