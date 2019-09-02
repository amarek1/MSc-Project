import pandas as pd
from sklearn.model_selection import train_test_split
from dtreeplt import dtreeplt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from global_functions import get_balanced_data, get_model_performance
from global_functions import plot_confusion_matrix, cm_analysis
from keras.models import load_model
from sklearn.preprocessing import RobustScaler
np.random.seed(7)

# load the data
file_name = 'data/bioresponse/bio_clean.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

# unbalanced data
X = data.drop('class',axis=1)
y = data['class']

X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = train_test_split(X, y, test_size=0.25,
                                                                                                random_state=1)
scaler = RobustScaler()
scaler.fit(X_train_unbalanced)
X_train_unbalanced = scaler.transform(X_train_unbalanced)
X_test_unbalanced = scaler.transform(X_test_unbalanced)

# balanced data
# even out the data set -> 1:1 ratio of fraud and non fraud

X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = get_balanced_data(data)

scaler = RobustScaler()
scaler.fit(X_train_balanced)
X_train_balanced = scaler.transform(X_train_balanced)
X_test_balanced = scaler.transform(X_test_balanced)

# unpack unbalanced model
path1 = '1) classification algorithms/neural networks/bioresponse/model_NN_unbalanced_bio.pkl'
# with open(path, 'rb') as file:
#     unbalanced_model = pickle.load(file)
unbalanced_model = load_model(path1)

# unpack balanced model
path2 = '1) classification algorithms/neural networks/bioresponse/model_NN_balanced_bio.pkl'
# with open(path, 'rb') as file:
#     balanced_model = pickle.load(file)
balanced_model = load_model(path2)
print('models unpacked')

# predict labels
unbalanced_predictions = unbalanced_model.predict(X_test_unbalanced)
unbalanced_predictions = unbalanced_predictions[:,1]
unbalanced_predictions = [int(round(x)) for x in unbalanced_predictions]

balanced_predictions = balanced_model.predict(X_test_balanced)
balanced_predictions = balanced_predictions[:,1]
balanced_predictions = [int(round(x)) for x in balanced_predictions]
print('labels predicted')


# print the confusion matrix, precision, recall, etc.
get_model_performance(unbalanced_model, 'unbalanced', X_test_unbalanced, y_test_unbalanced, 'NN','bioresponse dataset')
plt.savefig('1) classification algorithms/assess model performance/bioresponse/figures/PRcurve_nn_unbalanced_bio_short.png')
plt.close()
get_model_performance(balanced_model, 'balanced', X_test_balanced, y_test_balanced, 'NN','bioresponse dataset')
plt.savefig('1) classification algorithms/assess model performance/bioresponse/figures/PRcurve_nn_balanced_bio_short.png')
plt.close()
print('PR curve ready')

cm_analysis(y_test_balanced,balanced_predictions,filename='1) classification algorithms/assess model performance/bioresponse/figures/cm_nn_balanced_bio_short.png',labels=[0, 1],
            ymap=['response','no response'],title='NN performance on balanced data\nbioresponse dataset')

cm_analysis(y_test_unbalanced,unbalanced_predictions,filename='1) classification algorithms/assess model performance/bioresponse/figures/cm_nn_unbalanced_bio_short.png',labels=[0, 1],
            ymap=['response','no response'],title='NN performance on unbalanced data\nbioresponse dataset')
print('confusion matrix ready')
