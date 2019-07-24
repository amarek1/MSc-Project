import pandas as pd
from sklearn.model_selection import train_test_split
from dtreeplt import dtreeplt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, classification_report
from global_functions import get_model_performance
from global_functions import plot_confusion_matrix, cm_analysis
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from global_functions import get_balanced_data
from sklearn.metrics import classification_report
np.random.seed(7)

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

# balanced data
# even out the data set -> 1:1 ratio of fraud and non fraud
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = get_balanced_data(data)


fraud_data_size = [0,100,200]#,300,400,492]
d={}
report_dict={}
for i in range(0, len(fraud_data_size)):
    path = '4) final figures/recall improvement/models/' + 'ori fraud only/model_forest_unbalanced_ori_fraud_' + str(
        fraud_data_size[i]) + '.pkl'
    with open(path, 'rb') as file:
        d['model_'+str(fraud_data_size[i])] = pickle.load(file)
        model_predictions = d['model_0'].predict(X_test_balanced)
        model_predictions = [int(round(x)) for x in model_predictions]
        report = classification_report(y_test_balanced, model_predictions, labels=None,
                                       target_names=['normal', 'fraud'], digits=2, output_dict=True)
        report_dict['model_'+str(fraud_data_size[i])]=report
print(report_dict['model_100']['fraud']['precision'])










# cm_analysis(y_test_balanced,model_0_predictions,filename='4) final figures/cm_rf_balanced_ori',labels=[0, 1],
#             ymap=['normal','fraud'],title='RF performance on balanced data')

