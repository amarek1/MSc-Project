# import importlib.util as import_functions
# spec = import_functions.spec_from_file_location('global_functions.py',
#                                               'C:/Users/amarek/PycharmProjects/data_lab_clean/global_functions.py')
# catch = import_functions.module_from_spec(spec)
# spec.loader.exec_module(catch)

import pandas as pd
from sklearn.model_selection import train_test_split

# create train and test data with equal number of classes
def get_balanced_data(data):

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, data['class'], test_size=0.2, random_state=1)

    # even out the data set -> 1:1 ratio of 0 and 1 classes
    data_training = train_data.sample(frac=1)  # shuffle
    data_testing = test_data.sample(frac=1) # shuffle

    fraud_data_training = data_training.loc[data_training['class'] == 1]
    fraud_data_testing = data_testing.loc[data_testing['class'] == 1]

    non_fraud_data_training = data_training.loc[data_training['class'] == 0][:len(fraud_data_training)]
    non_fraud_data_testing = data_testing.loc[data_testing['class'] == 0][:len(fraud_data_testing)]

    even_data_training = pd.concat([fraud_data_training, non_fraud_data_training])
    even_data_testing = pd.concat([fraud_data_testing, non_fraud_data_testing])

    even_data_training = even_data_training.sample(frac=1, random_state=42)
    even_data_testing = even_data_testing.sample(frac=1, random_state=42)

    train_data = even_data_training.drop('class', axis=1)
    test_data = even_data_testing.drop('class', axis=1)
    train_labels = even_data_training['class']
    test_labels = even_data_testing['class']

    return train_data, test_data, train_labels, test_labels
