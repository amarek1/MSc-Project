import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from global_functions import get_balanced_data
import numpy
numpy.random.seed(7)

# create train and test data with equal number of classes
def get_balanced_data(data):
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_labels, test_labels = \
        train_test_split(data, data['class'], test_size=0.25, random_state=1)

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

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

X_train, X_test, y_train, y_test = get_balanced_data(data)

def get_NN_model(data=data, lr=0.001, loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                 validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2,
                 model_name='model_NN_balanced.pkl'):

    # create the neural net
    n_inputs = X_train.shape[1]
    balanced_model = Sequential()
    balanced_model.add(Dense(n_inputs, input_dim=n_inputs, activation='relu'))
    balanced_model.add(Dense(32, activation='relu'))
    balanced_model.add(Dense(2, activation='softmax'))
    # model.summary()

    # compile the model
    balanced_model.compile(Adam(lr=lr), loss=loss, metrics=metrics)
    balanced_model.fit(X_train, y_train, validation_split=validation_split, batch_size=batch_size,
                               epochs=epochs, shuffle=shuffle, verbose=verbose)

    path = '1) classification algorithms/neural networks/credit card fraud/'+model_name

    balanced_model.save(path)

    # with open(path, 'wb') as file:
    #     pickle.dump(model, file)
    # return


get_NN_model()