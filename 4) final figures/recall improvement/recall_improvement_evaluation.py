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
import matplotlib.pyplot as plt
np.random.seed(7)

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
data = pd.read_pickle(file_name)

# balanced data
# even out the data set -> 1:1 ratio of fraud and non fraud
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = get_balanced_data(data)


# ############## original fraud #############################
#
# fraud_data_size = [0,100,200,300,381]
# d={}
# report_dict_ori={}
# for i in range(0, len(fraud_data_size)):
#     path = '4) final figures/recall improvement/models/' + 'ori fraud only/model_forest_unbalanced_ori_fraud_' + str(
#         fraud_data_size[i]) + '.pkl'
#     with open(path, 'rb') as file:
#         d['model_'+str(fraud_data_size[i])] = pickle.load(file)
#         model_predictions = d['model_'+str(fraud_data_size[i])].predict(X_test_balanced)
#         model_predictions = [int(round(x)) for x in model_predictions]
#         report = classification_report(y_test_balanced, model_predictions, labels=None,
#                                        target_names=['normal', 'fraud'], digits=2, output_dict=True)
#         report_dict_ori['model_' + str(fraud_data_size[i])]=report
#
#
# plt.plot([0,100,200,300,381], [report_dict_ori['model_0']['fraud']['recall'], report_dict_ori['model_100']['fraud']['recall'] ,
#                                report_dict_ori['model_200']['fraud']['recall'], report_dict_ori['model_300']['fraud']['recall'],
#                                report_dict_ori['model_381']['fraud']['recall']],label='real fraud',marker='o')
# plt.ylabel('f1-score')
# plt.xlabel('# original fraud data')
# plt.title('The effect of # of fraud data examples on f1-score')
#
#
# ############## synthetic fraud #############################
#
# fraud_data_size = [0,100,200,300,381]
# d={}
# report_dict_syn={}
# for i in range(0, len(fraud_data_size)):
#     path = '4) final figures/recall improvement/models/' + 'syn fraud only/model_forest_unbalanced_syn_fraud_' + str(
#         fraud_data_size[i]) + '.pkl'
#     with open(path, 'rb') as file:
#         d['model_'+str(fraud_data_size[i])] = pickle.load(file)
#         model_predictions = d['model_'+str(fraud_data_size[i])].predict(X_test_balanced)
#         model_predictions = [int(round(x)) for x in model_predictions]
#         report = classification_report(y_test_balanced, model_predictions, labels=None,
#                                        target_names=['normal', 'fraud'], digits=2, output_dict=True)
#         report_dict_syn['model_' + str(fraud_data_size[i])]=report
#
#
# plt.plot([0,100,200,300,381], [report_dict_syn['model_0']['fraud']['recall'], report_dict_syn['model_100']['fraud']['recall'] ,
#                                report_dict_syn['model_200']['fraud']['recall'], report_dict_syn['model_300']['fraud']['recall'],
#                                report_dict_syn['model_381']['fraud']['recall']],label='synthetic fraud',marker='o')
# plt.ylabel('recall')
# plt.xlabel('# fraud data')
# plt.title('The effect of introducing fraud training data on recall')
# plt.grid()
# plt.legend()
# plt.yticks(np.arange(0,1,0.1))
# plt.savefig('4) final figures/recall improvement/effect of introducing fraud data on recall.png')


############# synthetic on top of original fraud #############################

# def plot_this(score='f1-score'):
#     fraud_data_size = [0,100,200,300,400,500,1000,2000,3000,4000,5000]
#     d={}
#     report_dict_mix={}
#     for i in range(0, len(fraud_data_size)):
#         path = '4) final figures/recall improvement/models/' + 'ori and syn fraud/model_forest_unbalanced_mix_fraud_' + str(
#             fraud_data_size[i]) + '.pkl'
#         with open(path, 'rb') as file:
#             d['model_'+str(fraud_data_size[i])] = pickle.load(file)
#             model_predictions = d['model_'+str(fraud_data_size[i])].predict(X_test_balanced)
#             model_predictions = [int(round(x)) for x in model_predictions]
#             report = classification_report(y_test_balanced, model_predictions, labels=None,
#                                            target_names=['normal', 'fraud'], digits=2, output_dict=True)
#             report_dict_mix['model_' + str(fraud_data_size[i])]=report
#
#
#     plt.plot([0,100,200,300,400,500,1000,2000,3000,4000,5000],[report_dict_mix['model_0']['fraud'][score],report_dict_mix['model_100']['fraud'][score],
#                           report_dict_mix['model_200']['fraud'][score],report_dict_mix['model_300']['fraud'][score],
#                                   report_dict_mix['model_400']['fraud'][score],
#                                       report_dict_mix['model_500']['fraud'][score],report_dict_mix['model_1000']['fraud'][score],report_dict_mix['model_2000']['fraud'][score],
#                           report_dict_mix['model_3000']['fraud'][score],report_dict_mix['model_4000']['fraud'][score],
#                                   report_dict_mix['model_5000']['fraud'][score]],marker='o')
#     plt.axhline(y=report_dict_mix['model_0']['fraud'][score], color='r', linestyle='--')
#     plt.ylabel(score)
#     plt.xlabel('# synthetic fraud data')
#     plt.title('The effect of adding synthetic data to original training data on '+str(score))
#     #plt.yticks(np.arange(0.7, 0.801, 0.01))
#     plt.grid()
#     plt.savefig('4) final figures/recall improvement/effect of adding fraud data on '+str(score)+'2.png')
#     plt.show()
#     return

def plot_this(score='f1-score'):
    fraud_data_size = [0,100,200,300,400,500,1000,2000,3000,4000,5000]
    d={}
    report_dict_mix={}
    for i in range(0, len(fraud_data_size)):
        path = '4) final figures/recall improvement/models/' + 'ori and syn fraud/model_forest_balanced_mix_fraud_' + str(
            fraud_data_size[i]) + '.pkl'
        with open(path, 'rb') as file:
            d['model_'+str(fraud_data_size[i])] = pickle.load(file)
            model_predictions = d['model_'+str(fraud_data_size[i])].predict(X_test_balanced)
            model_predictions = [int(round(x)) for x in model_predictions]
            report = classification_report(y_test_balanced, model_predictions, labels=None,
                                           target_names=['normal', 'fraud'], digits=2, output_dict=True)
            report_dict_mix['model_' + str(fraud_data_size[i])]=report


    plt.plot([0,100,200,300,400,500,1000,2000,3000,4000,5000],[report_dict_mix['model_0']['fraud'][score],report_dict_mix['model_100']['fraud'][score],
                          report_dict_mix['model_200']['fraud'][score],report_dict_mix['model_300']['fraud'][score],
                                  report_dict_mix['model_400']['fraud'][score],
                                      report_dict_mix['model_500']['fraud'][score],report_dict_mix['model_1000']['fraud'][score],report_dict_mix['model_2000']['fraud'][score],
                          report_dict_mix['model_3000']['fraud'][score],report_dict_mix['model_4000']['fraud'][score],
                                  report_dict_mix['model_5000']['fraud'][score]],marker='o')
    plt.axhline(y=report_dict_mix['model_0']['fraud'][score], color='r', linestyle='--')
    plt.ylabel(score)
    plt.xlabel('# synthetic fraud data')
    plt.title('The effect of adding synthetic data to original training data on '+str(score))
    #plt.yticks(np.arange(0.7, 0.801, 0.01))
    plt.grid()
    plt.savefig('4) final figures/recall improvement/balanced effect of adding fraud data on '+str(score)+'2.png')
    plt.show()
    return

plot_this(score='recall')

