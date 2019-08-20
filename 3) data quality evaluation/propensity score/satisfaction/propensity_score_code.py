import pandas as pd
import numpy as np
import importlib.util
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn import svm
np.random.seed(1)

file_name = 'data/satisfaction/satisfaction clean_scaled.pkl'
real_data = pd.read_pickle(file_name)
real_fraud = real_data.loc[real_data['class'] == 1]

# load the synthetic data
file_name = '2) synthetic data generation/GAN/satisfaction/GAN_sat_5000.pkl'
syn_fraud = pd.read_pickle(file_name)
syn_fraud = syn_fraud[:len(real_fraud)]



def add_labels(real_data, synthetic_data):

    # add labels 0 for real and 1 for synthetic
    data = pd.concat([real_data, synthetic_data], ignore_index=True)
    o_labels = np.zeros((len(real_data)), dtype=int)
    s_labels = np.ones((len(synthetic_data)), dtype=int)
    labels = np.concatenate([o_labels, s_labels], axis=0)
    data['class'] = labels
    x = data.drop('class', axis=1)
    y = data['class']

    return x, y


X, Y = add_labels(real_fraud, syn_fraud)

cv_5 = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
scoring = 'accuracy'


def baseline_model(optimizer='adam', learn_rate=0.1):
    model = Sequential()
    model.add(Dense(100, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))  # 8 is the dim/ the number of hidden units (units are the kernel)
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_best_hyperparameters(X,Y):
    keras = KerasClassifier(build_fn=baseline_model, verbose=0)
    batch_size = [32, 64]
    epochs = [50, 100]
    learn_rate = [0.1, 0.01]
    optimizer = ['Adam']

    kerasparams = dict(batch_size=batch_size, learn_rate=learn_rate, optimizer=optimizer, epochs=epochs)

    a = GridSearchCV(estimator=keras, param_grid=kerasparams, cv=cv_5, iid=False, n_jobs=4, verbose=2)
    a.fit(X, Y)
    print(a.best_params_)
    print(a.best_score_)

    cv_1 = cross_val_score(a, X, Y, cv=cv_5, scoring=scoring, verbose=2, n_jobs=4)
    print('cross_val_score:', cv_1)

def get_probability_labels(x, y):
    all_predictions = []
    estimator = KerasClassifier(batch_size=32, epochs=100, optimizer='Adam', build_fn=baseline_model, verbose=0)
    for train_index, test_index in cv_5.split(x, y):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        estimator.fit(X_train, y_train)
        predictions = estimator.predict_proba(X_test)
        predictions = list(predictions[:, 1])
        all_predictions.append(predictions)
        a = [j for i in all_predictions for j in i] #remove nested list
    return a


def get_propensity_score(probability_labels, synthetic_data, real_data):

    divide_by_n = 1/len(probability_labels)
    proportion_of_syn = len(synthetic_data)/(len(synthetic_data) + len(real_data))

    sum_p_c = 0
    for i in range(0, len(probability_labels)):
        p_minus_c_squared = (probability_labels[i] - proportion_of_syn) ** 2
        sum_p_c = sum_p_c + p_minus_c_squared

    propensity_score = divide_by_n * sum_p_c
    return propensity_score


# get_best_hyperparameters(X, Y)
probability_labels = get_probability_labels(X, Y)
the_score = get_propensity_score(probability_labels, syn_fraud, real_fraud)
print(the_score)  # the lower the score the better
