import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from global_functions import get_balanced_data
import numpy
numpy.random.seed(7)

# load the data
file_name = 'data/bioresponse/bio_clean.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


def get_NN_model(lr=0.001, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                 validation_split=0.25, batch_size=25, epochs=20, shuffle=True, verbose=2,
                 model_name='model_NN_unbalanced_bio.pkl'):

    # create the neural net
    n_inputs = X_train.shape[1]
    model = Sequential()
    model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # model.summary()

    # compile the model
    model.compile(Adam(lr=lr), loss=loss, metrics=metrics)
    model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size,
                               epochs=epochs, shuffle=shuffle, verbose=verbose)

    path = '1) classification algorithms/neural networks/bioresponse/'+model_name

    model.save(path)

    # with open(path, 'wb') as file:
    #     pickle.dump(model, file)
    # return


get_NN_model()
