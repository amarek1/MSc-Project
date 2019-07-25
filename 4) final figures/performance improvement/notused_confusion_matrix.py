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
np.random.seed(7)

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

# # unbalanced data
# X = data.drop('class',axis=1)
# y = data['class']
# X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = train_test_split(X, y, test_size=0.2,
#                                                                                                 random_state=1)

# balanced data
# even out the data set -> 1:1 ratio of fraud and non fraud
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = get_balanced_data(data)

# unpack unbalanced model
path = '4) final figures/recall improvement/models/ori and syn fraud/model_forest_unbalanced_mix_fraud_0.pkl'
with open(path, 'rb') as file:
    model_0 = pickle.load(file)


# unpack balanced model
path = '4) final figures/recall improvement/models/ori and syn fraud/model_forest_unbalanced_mix_fraud_5000.pkl'
with open(path, 'rb') as file:
    model_5000 = pickle.load(file)

# predict labels
model0_predictions = model_0.predict(X_test_balanced)
model0_predictions = [int(round(x)) for x in model0_predictions]
model5000_predictions = model_5000.predict(X_test_balanced)
model5000_predictions = [int(round(x)) for x in model5000_predictions]

# print the confusion matrix, precision, recall, etc.
get_model_performance(model_0, 'unbalanced', X_test_balanced, y_test_balanced)
get_model_performance(model_5000, 'balanced', X_test_balanced, y_test_balanced)


cm_analysis(y_test_balanced, model0_predictions, filename='4) final figures/recall improvement/cm_0', labels=[0, 1],
            ymap=['normal','fraud'], title='Model trained on real data')

cm_analysis(y_test_balanced, model5000_predictions, filename='4) final figures/recall improvement/cm_5000', labels=[0, 1],
            ymap=['normal','fraud'], title='Real + 5000 fraud')
