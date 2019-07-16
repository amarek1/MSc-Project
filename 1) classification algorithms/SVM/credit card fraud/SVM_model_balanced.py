# Load libraries
import pandas as pd
import pickle
import numpy
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from global_functions import get_balanced_data
numpy.random.seed(7)

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

# # cross-validation and parameter optimisation
# parameters = [
#   {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
#  ]
# clf = GridSearchCV(svm.SVC(), parameters, n_jobs=6, cv=4, scoring='f1', verbose=2)
# clf.fit(X=X_train, y=y_train)
# SCM_model = clf.best_estimator_
# print(clf.best_score_, clf.best_params_)

def get_SVM_model(data=data, kernel='rbf', gamma=0.0001, C=1000, probability=True, model_name='model_SVM_balanced.pkl'):

    X_train, X_test, y_train, y_test = get_balanced_data(data)

    # Create SVM classifer object
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C, probability=probability)

    # Train SVM classifer
    model = clf.fit(X_train, y_train)

    path = '1) classification algorithms/SVM/credit card fraud/'+model_name

    with open(path, 'wb') as file:
        pickle.dump(model, file)
    return

get_SVM_model()
