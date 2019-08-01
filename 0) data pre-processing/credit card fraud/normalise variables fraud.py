# Load libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


# load the data
file_name = 'C:/Users/amarek/PycharmProjects/data_lab/creditcard.csv'
data = pd.read_csv(file_name)

# rescale time and amount to be similar to features
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))
data['class'] = data['Class']

data.drop(['Time','Amount','Class'], axis=1, inplace=True)


data.to_csv(r'C:/Users/amarek/PycharmProjects/data_lab/creditcard_normalised.csv', index = None)

print('success')