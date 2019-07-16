import pandas as pd
from sklearn.model_selection import train_test_split
from dtreeplt import dtreeplt
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from global_functions import get_balanced_data, get_model_performance
np.random.seed(7)

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

# unbalanced data
X = data.drop('class',axis=1)
y = data['class']

X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = train_test_split(X, y, test_size=0.2,
                                                                                                random_state=1)

# balanced data
# even out the data set -> 1:1 ratio of fraud and non fraud

X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = get_balanced_data(data)

#111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111unbalanced
# unpack unbalanced model
path = '1) classification algorithms/SVM/credit card fraud/model_SVM_balanced.pkl'
with open(path, 'rb') as file:
    unbalanced_model = pickle.load(file)

# unpack balanced model
path = '1) classification algorithms/SVM/credit card fraud/model_SVM_balanced.pkl'
with open(path, 'rb') as file:
    balanced_model = pickle.load(file)

# predict labels
unbalanced_predictions = unbalanced_model.predict(X_test_unbalanced)
balanced_predictions = balanced_model.predict(X_test_balanced)
#
# cm_balanced = confusion_matrix(y_test_balanced, balanced_predictions)
# cm_unbalanced = confusion_matrix(y_test_unbalanced, unbalanced_predictions)
#
# print(cm_balanced)
# print(cm_unbalanced)
# print('balanced model:',classification_report(y_test_balanced, balanced_predictions))
# print('unbalanced model:',classification_report(y_test_unbalanced, unbalanced_predictions))

get_model_performance(unbalanced_model, 'unbalanced', X_test_unbalanced, y_test_unbalanced)
get_model_performance(balanced_model, 'balanced', X_test_balanced, y_test_balanced)

def get_confusion_matrix(y_true, y_pred, labels, title):
    axs = plt.subplots(2)
    for i in range(0,len(y_pred)):
        axs[i] = plt.subplot()
        cm = confusion_matrix(y_true, y_pred[i], labels)
        sns.heatmap(cm, annot=True, ax = axs, cmap="Blues") #annot=True to annotate cells
        axs[i].set_xlabel('Predicted labels')
        axs[i].set_ylabel('True labels')
        axs[i].set_title(title)
        axs[i].xaxis.set_ticklabels(['fraud', 'normal'])
        axs[i].yaxis.set_ticklabels(['fraud', 'normal'])
    plt.show()

get_confusion_matrix(y_test_balanced, balanced_predictions,[0,1],'blabla')

axs = plt.subplots(2)
cm = confusion_matrix(y_test_balanced, a, labels)
axs[1] = sns.heatmap(cm, annot=True, ax = axs, cmap="Blues")