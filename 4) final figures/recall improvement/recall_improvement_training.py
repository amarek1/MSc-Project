# the same model parameters as for original balanced model
# {'bootstrap': True, 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100}

#the same model parameters as for original balanced model
#
import pandas as pd
from sklearn.model_selection import train_test_split
from dtreeplt import dtreeplt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from global_functions import get_model_performance
from global_functions import plot_confusion_matrix, cm_analysis
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
np.random.seed(7)

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

def get_balanced_data(data, len_training_fraud, len_training_original):

    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, data['class'], test_size=0.25, random_state=1)

    # even out the data set -> 1:1 ratio of 0 and 1 classes
    data_training = train_data.sample(frac=1)  # shuffle
    data_testing = test_data.sample(frac=1) # shuffle

    fraud_data_training = data_training.loc[data_training['class'] == 1][:len_training_fraud]
    fraud_data_testing = data_testing.loc[data_testing['class'] == 1]

    non_fraud_data_training = data_training.loc[data_training['class'] == 0][:len_training_original]
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


def get_forest_model(data=data, balanced=True, len_training_fraud=0, len_training_original=200000,
                     model_name='ori fraud only/model_forest_unbalanced_ori_syn_'+'a'+'.pkl'):

    if balanced == True:
        X_train, X_test, y_train, y_test = get_balanced_data(data, len_training_fraud, len_training_original)
        optimized_model = RandomForestRegressor(bootstrap=True, max_depth=10, max_features='auto',
                                      min_samples_split=20, n_estimators=100)
    else:
        X = data.drop('class', axis=1)
        y = data['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        optimized_model = RandomForestRegressor(bootstrap=True, max_depth=10, max_features='auto',
                                      min_samples_split=20, n_estimators=100)


    model = optimized_model.fit(X_train, y_train)

    path = '4) final figures/recall improvement/models/' + model_name
    with open(path, 'wb') as file:
        pickle.dump(model, file)

    return

#get_forest_model()

fraud_data_size = [0,100,200,300,400,492]
for i in range(0, len(fraud_data_size)):
    get_forest_model(data=data, balanced=True, len_training_fraud=fraud_data_size[i], len_training_original=200000,
                     model_name='ori fraud only/model_forest_unbalanced_ori_fraud_'+str(fraud_data_size[i])+'.pkl')
