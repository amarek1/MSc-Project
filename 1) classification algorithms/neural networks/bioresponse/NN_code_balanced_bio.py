import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from global_functions import get_balanced_data
import numpy
numpy.random.seed(7)

# load the data
file_name = 'data/bioresponse/bio_short.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

X_train, X_test, y_train, y_test = get_balanced_data(data)


def get_NN_model(data=data, lr=0.001, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                 validation_split=0.25, batch_size=25, epochs=20, shuffle=True, verbose=2,
                 model_name='model_NN_balanced_bio_short.pkl'):

    # create the neural net
    n_inputs = X_train.shape[1]
    balanced_model = Sequential()
    balanced_model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    balanced_model.add(Dense(32, activation='relu'))
    balanced_model.add(Dense(64, activation='relu'))
    balanced_model.add(Dense(32, activation='relu'))
    balanced_model.add(Dense(2, activation='softmax'))
    # model.summary()

    # compile the model
    balanced_model.compile(Adam(lr=lr), loss=loss, metrics=metrics)
    balanced_model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size,
                               epochs=epochs, shuffle=shuffle, verbose=verbose)

    path = '1) classification algorithms/neural networks/bioresponse/'+model_name

    balanced_model.save(path)

    # with open(path, 'wb') as file:
    #     pickle.dump(model, file)
    # return


get_NN_model()