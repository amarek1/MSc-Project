# make sure that column names in the original data are in the same format as synthetic
import pandas as pd

file_n = 'data/customer churn/customer churn modified.pkl'
real_data = pd.read_pickle(file_n)
real_cols = real_data.columns
print('real: ',len(real_cols))

# load the data
file_name = '2) synthetic data generation/tGAN/customer churn/normal/tGAN_normal_5174.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)
# data = data.drop(columns=['Unnamed: 0', 'X'])
# fraud_data = data.loc[data['class'] == 1]
# normal_data = data.loc[data['class'] == 0]
data_cols = data.columns
print('syn: ',len(data_cols))
print('synthetic: ',data.columns, '\nreal:', real_data.columns,'\n\n')

#put columns in the same order
data = data[['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
       'class', 'gender_Female',
       'gender_Male', 'MultipleLines_No', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']]

#make sure that the names are formatted in the same way
data.columns = real_data.columns
print('synthetic: ',data.columns, '\nreal:', real_data.columns)
data.to_pickle(file_name)
