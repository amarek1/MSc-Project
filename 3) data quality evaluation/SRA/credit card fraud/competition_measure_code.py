import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import importlib.util
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import importlib.util as import_functions
from global_functions import get_balanced_data
np.random.seed(1)


file_name = 'data/credit card fraud/data_creditcard.pkl'
real_data = pd.read_pickle(file_name)


file_name = '2) synthetic data generation/tGAN/credit card fraud/tGAN_normal_284315.pkl'
syn_data_normal = pd.read_pickle(file_name)
syn_data_normal = syn_data_normal.loc[syn_data_normal['class'] == 0]


file_name = '2) synthetic data generation/tGAN/credit card fraud/tGAN_fraud_5000.pkl'
syn_data_fraud = pd.read_pickle(file_name)
syn_data_fraud = syn_data_fraud[:492]

syn_data = pd.concat([syn_data_normal, syn_data_fraud], sort=False)



def get_accuracies(data):

    X_train, X_test, y_train, y_test = get_balanced_data(data)

    seed = 1
    rfc = RandomForestClassifier(bootstrap=True, max_depth=10, max_features='auto', min_samples_leaf=2,
                                 min_samples_split=10,
                                 n_estimators=500)

    rfc2 = RandomForestClassifier(bootstrap=False, max_depth=2, max_features='auto', min_samples_leaf=5,
                                 min_samples_split=20,
                                 n_estimators=100)

    gbm = GradientBoostingClassifier(min_samples_split=25, min_samples_leaf=25, loss='deviance', learning_rate=0.1,
                                     max_depth=5, max_features='auto', criterion='friedman_mse', n_estimators=100)


    def baseline_model(optimizer='adam', learn_rate=0.01):
        model = Sequential()
        model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(50, activation='relu'))  # 8 is the dim/ the number of hidden units (units are the kernel)
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    keras = KerasClassifier(build_fn=baseline_model, batch_size=32, epochs=100, verbose=0, optimizer='Adam')

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    svm = SVC(gamma="scale", probability=True, kernel='rbf', C=0.5)

    models = [('GBM', gbm), ('RFC', rfc), ('RFC2', rfc2), ('Keras', keras), ('SVM', svm)]

    results = []
    names = []
    scoring = 'accuracy'

    accuracy = []
    for name, model in models:
        cv_results = cross_val_score(model, X_train, y_train, cv=outer_cv, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        # msg = "Cross-validation Accuracy %s: %f (+/- %f )" % (name, cv_results.mean() * 100, cv_results.std() * 100)
        # print(msg)
        model.fit(X_train, y_train)
        # print('Test set accuracy: {:.2f}'.format(model.score(X_test, y_test) * 100), '%')
        # accuracy.append(name)
        accuracy.append(model.score(X_test, y_test))
    return accuracy


real_accuracy = get_accuracies(real_data)
syn_accuracy = get_accuracies(syn_data)
print('accuracy on real data:', real_accuracy, 'accuracy on synthetic data:', syn_accuracy)

# real_accuracy = [1,2,1,4,5]
# syn_accuracy = [1,2,3,4,5]


def SRA(real_score, syn_score):
    k = len(real_score)
    a = 1/(k*(k-1))

    values = []
    for i in range(0, len(real_score)):
        for j in range(i+1, len(real_score)):
            b = (real_score[i] - real_score[j]) * (syn_score[i] - syn_score[j])
            c = (real_score[j] - real_score[i]) * (syn_score[j] - syn_score[i])
            values.append(b)
            values.append(c)

    values2 = []
    for i in range(0, len(values)):
        if values[i] > 0:
            values2.append(1)

    return a*sum(values2)


SRA_score = SRA(real_accuracy, syn_accuracy)
print('SRA:', SRA_score)  # the closer to 1, the better
