import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from global_functions import get_balanced_data
np.random.seed(7)


def get_forest_model(dataset_name='data_creditcard.pkl', balanced=False):
    # load the data
    file_name = 'data/credit card fraud/'+dataset_name  # set working directory to MSc Project
    data = pd.read_pickle(file_name)

    if balanced == True:
        X_train, X_test, y_train, y_test = get_balanced_data(data)
    else:
        X = data.drop('class', axis=1)
        y = data['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_forest_model(balanced=True)


# perform grid search to find the best parameters
def get_best_params(train_data, train_labels):
    # Number of trees in random forest
    n_estimators = [10, 100]  # [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    #max_features = ['auto']  # ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [3, 5, 10]  # [int(x) for x in np.linspace(10, 110, num = 11)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 10, 20]
    #criterion = ['gini', 'entropy']
    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   # 'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   # 'min_samples_leaf': min_samples_leaf,
                   #'criterion': criterion,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = GridSearchCV(estimator=rf, param_grid=random_grid, cv=3, verbose=2, n_jobs=1)
    rf_random.fit(train_data, train_labels)
    print(rf_random.best_params_)

best_params = get_best_params(X_train[0:50], y_train[0:50])
print(best_params[1])


# optimized_model = RandomForestRegressor(bootstrap=True, max_depth=5, max_features='auto',
#                                          min_samples_split=2, n_estimators=100)
#
# # Fit the random search model
# model = optimized_model.fit(X_train, y_train)
#
# # path for saving the model
# path = 'C:/Users/amarek/PycharmProjects/data_lab/models/decision tree/forest_model_bal_ori.pkl'
#
# # save the model
# with open(path, 'wb') as file:
#     pickle.dump(model, file)
