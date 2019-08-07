# the same model parameters as for original balanced model
# {'bootstrap': True, 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100}
# the same model parameters as for original balanced model

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
np.random.seed(7)

###################### load the data ####################################

# load the real data
file_name = 'data/customer churn/customer churn modified.pkl'  # set working directory to MSc Project
real_data = pd.read_pickle(file_name)

# # load synthetic fraud examples
# file_name = '2) synthetic data generation/WcGAN/credit card fraud/WcGAN results/WcGAN_fraud_492_Adam.pkl'
# synthetic_fraud = pd.read_pickle(file_name)

train_data, test_data, train_labels, test_labels = train_test_split(real_data, real_data['class'], test_size=0.25, random_state=1)

syn1 = train_data.loc[train_data['class']==1]
syn2 = train_data.loc[train_data['class']==1]
# syn3 = train_data.loc[train_data['class']==1]
# syn4 = train_data.loc[train_data['class']==1]
# syn5 = train_data.loc[train_data['class']==1]
# syn6 = train_data.loc[train_data['class']==1]
# syn7 = train_data.loc[train_data['class']==1]
# syn8 = train_data.loc[train_data['class']==1]
# syn9 = train_data.loc[train_data['class']==1]
# syn10 = train_data.loc[train_data['class']==1]
# syn11 = train_data.loc[train_data['class']==1]
# syn12 = train_data.loc[train_data['class']==1]
# syn13 = train_data.loc[train_data['class']==1]
# syn14 = train_data.loc[train_data['class']==1]


synthetic_fraud = pd.concat([syn1,syn2])  # ,syn3,syn4,syn5,syn6,syn7,syn8,syn9,syn10,syn11,syn12,syn13,syn14])


####################### functions ######################################


def get_data(real_data, synthetic_data, nr_normal_training, nr_fraud_training, nr_synthetic_fraud_training, test_size):

    train_data, test_data, train_labels, test_labels = \
        train_test_split(real_data, real_data['class'], test_size=test_size, random_state=1)

    # even out the data set -> 1:1 ratio of 0 and 1 classes
    data_training = train_data.sample(frac=1)  # shuffle
    data_testing = test_data.sample(frac=1) # shuffle

    fraud_data_training = data_training.loc[data_training['class'] == 1][:nr_fraud_training]
    fraud_data_testing = data_testing.loc[data_testing['class'] == 1]

    non_fraud_data_training = data_training.loc[data_training['class'] == 0][:nr_normal_training]
    non_fraud_data_testing = data_testing.loc[data_testing['class'] == 0][:len(fraud_data_testing)]

    synthetic_data = synthetic_data.sample(frac=1)
    synthetic_fraud = synthetic_data.loc[synthetic_data['class']==1][:nr_synthetic_fraud_training]

    fraud_data_training = pd.concat([fraud_data_training,synthetic_fraud])

    even_data_training = pd.concat([fraud_data_training, non_fraud_data_training])
    even_data_testing = pd.concat([fraud_data_testing, non_fraud_data_testing])

    even_data_training = even_data_training.sample(frac=1, random_state=42)
    even_data_testing = even_data_testing.sample(frac=1, random_state=42)

    train_data = even_data_training.drop('class', axis=1)
    test_data = even_data_testing.drop('class', axis=1)
    train_labels = even_data_training['class']
    test_labels = even_data_testing['class']

    return train_data, test_data, train_labels, test_labels


def get_forest_model(real_data=real_data, synthetic_data=synthetic_fraud, folder='real fraud only', model_name='m1',
                     model_type='rf', nr_normal_training=213224, nr_fraud_training=0, nr_synthetic_fraud_training=0,
                     test_size=0.25):

    X_train, X_test, y_train, y_test = get_data(real_data, synthetic_data, nr_normal_training, nr_fraud_training,
                                                nr_synthetic_fraud_training, test_size)
    optimized_model = RandomForestRegressor(bootstrap=True, max_depth=10, max_features='auto', min_samples_split=20,
                                            n_estimators=100)
    model = optimized_model.fit(X_train, y_train)


    model_name = model_name +'_'+ model_type +'_'+ str(nr_normal_training) + '_'+str(nr_fraud_training) + '_' + \
                 str(nr_synthetic_fraud_training) +'_ts'+ str(test_size) + '.pkl'
    path = '4) final figures/performance improvement/models/' + folder + '/' + model_name
    with open(path, 'wb') as file:
        pickle.dump(model, file)

    return

####################### get models ######################################

nr_normal_training = [3847, 3847, 3847, 3847, 3847, 3847]
nr_fraud_training = [1435, 1435, 1435, 1435, 1435, 1435]
nr_synthetic_fraud_training = [0, 500, 1000, 1500, 2000, 2500]
for i in range(0, len(nr_normal_training)):
    get_forest_model(real_data=real_data, synthetic_data=synthetic_fraud, folder='control duplicate churn', model_name='rf',
                     model_type='real_duplicated', nr_normal_training=nr_normal_training[i], nr_fraud_training=nr_fraud_training[i],
                     nr_synthetic_fraud_training=nr_synthetic_fraud_training[i], test_size=0.25)

# nr_normal_training = [213224, 213224, 213224, 213224, 213224]
# nr_fraud_training = [0, 100, 200, 300, 381]
# nr_synthetic_fraud_training = [0,0,0,0,0]
# for i in range(0, len(nr_normal_training)):
#     get_forest_model(real_data=real_data, synthetic_data=synthetic_fraud, folder='real fraud only', model_name='m1',
#                      model_type='rf', nr_normal_training=nr_normal_training[i], nr_fraud_training=nr_fraud_training[i],
#                      nr_synthetic_fraud_training=nr_synthetic_fraud_training[i], test_size=0.25)
#
# nr_normal_training = [213224, 213224, 213224, 213224, 213224]
# nr_fraud_training = [0,0,0,0,0]
# nr_synthetic_fraud_training = [0, 100, 200, 300, 381]
# for i in range(0, len(nr_normal_training)):
#     get_forest_model(real_data=real_data, synthetic_data=synthetic_fraud, folder='syn fraud only', model_name='m1',
#                      model_type='rf', nr_normal_training=nr_normal_training[i], nr_fraud_training=nr_fraud_training[i],
#                      nr_synthetic_fraud_training=nr_synthetic_fraud_training[i], test_size=0.25)
#
# nr_normal_training = [213224, 213224, 213224, 213224, 213224, 213224, 213224, 213224, 213224, 213224, 213224]
# nr_fraud_training = [381,381,381,381,381,381,381,381,381,381,381]
# nr_synthetic_fraud_training = [0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
# for i in range(0, len(nr_normal_training)):
#     get_forest_model(real_data=real_data, synthetic_data=synthetic_fraud, folder='real and syn fraud', model_name='m1',
#                      model_type='rf', nr_normal_training=nr_normal_training[i], nr_fraud_training=nr_fraud_training[i],
#                      nr_synthetic_fraud_training=nr_synthetic_fraud_training[i], test_size=0.25)
#
# nr_normal_training = [381,481,581,681,781,881,1381,2381,3381,4381,5381]
# nr_fraud_training = [381,381,381,381,381,381,381,381,381,381,381]
# nr_synthetic_fraud_training = [0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
# for i in range(0, len(nr_normal_training)):
#     get_forest_model(real_data=real_data, synthetic_data=synthetic_fraud, folder='real and syn fraud', model_name='m1',
#                      model_type='rf', nr_normal_training=nr_normal_training[i], nr_fraud_training=nr_fraud_training[i],
#                      nr_synthetic_fraud_training=nr_synthetic_fraud_training[i], test_size=0.25)
#
# nr_normal_training = [2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000]
# nr_fraud_training = [381,381,381,381,381,381,381,381,381,381,381,381]
# nr_synthetic_fraud_training = [0, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000,10000,14760]
# for i in range(0, len(nr_normal_training)):
#     get_forest_model(real_data=real_data, synthetic_data=synthetic_fraud, folder='real and syn fraud', model_name='WcGAN_fraud_14760_Adam_l1',
#                      model_type='rf', nr_normal_training=nr_normal_training[i], nr_fraud_training=nr_fraud_training[i],
#                      nr_synthetic_fraud_training=nr_synthetic_fraud_training[i], test_size=0.25)

