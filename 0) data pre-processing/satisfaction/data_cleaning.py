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
data = data.drop(['ID'], axis=1)

# remove duplicate columns
remove_dup = []
cols = data.columns
for i in range(0,len(cols)-1):
    a = data[cols[i]].values
    for j in range(i+1, len(cols)):
        b = data[cols[j]].values
        if np.array_equal(a, b):
            remove_dup.append(cols[j])

# include columns with mostly zeros
remove_dup += ['num_var13_medio', 'saldo_medio_var13_medio_ult1']

data.drop(remove_dup, axis=1, inplace=True)

# check for NANs - no NaNs
data.columns[data.isna().any()].tolist()


# determine binary/categorical columns (0 or 1)
categorical = []
for col in data.columns:
    if data[col].max() == 1 and data[col].min() == 0:
        categorical.append(col)

# rename TARGET to class for consistency
data = data.rename(columns={'TARGET':'class'})

data.to_pickle('data/satisfaction/satisfaction clean.pkl')
data.to_csv('data/satisfaction/satisfaction clean.csv')

# seperate data into satisfied (0) and not satisfied (1)
normal_data = data[data['class']==0]
normal_data.to_csv('data/satisfaction/satisfaction normal.csv')

notsat_data = data[data['class']==1]
notsat_data.to_csv('data/satisfaction/satisfaction notsat.csv')
