# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
import pickle
import numpy
numpy.random.seed(7)
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import importlib.util as import_functions

# get access to global_functions
spec = import_functions.spec_from_file_location('global_functions.py',
                                              'C:/Users/amarek/PycharmProjects/MSc Project/global_functions.py')
catch = import_functions.module_from_spec(spec)
spec.loader.exec_module(catch)

# load the data
file_name = 'C:/Users/amarek/PycharmProjects/MSc Project/credit card/data/data_creditcard.pkl'
data = pd.read_pickle(file_name)

X_train, X_test, y_train, y_test = catch.get_balanced_data(data)

# # cross-validation and parameter optimisation
# parameters = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
# clf = GridSearchCV(svm.SVC(), parameters, n_jobs=4, cv=4, scoring='f1', verbose=2)
# clf.fit(X=X_train, y=y_train)
# SCM_model = clf.best_estimator_
# print(clf.best_score_, clf.best_params_)

# Create SVM classifer object
clf = svm.SVC(kernel='rbf',gamma=0.0001,C=100,probability=True)

# Train SVM Classifer
model = clf.fit(X_train,y_train)

path = 'C:/Users/amarek/PycharmProjects/credit card/data_lab_clean/models/SVM/SVM_model_balanced.pkl'

with open(path, 'wb') as file:
    pickle.dump(model, file)