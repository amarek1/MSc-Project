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
from global_functions import get_data
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
np.random.seed(7)

# load the data
file_name = 'data/credit card fraud/data_creditcard.pkl'
data = pd.read_pickle(file_name)

# get same train:test split as when training the model
# even out the data set -> 1:1 ratio of fraud and non fraud
X_train, X_test, y_train, y_test = get_data(real_data=data, synthetic_data=data,
                    nr_normal_training=381, nr_fraud_training=381, nr_synthetic_fraud_training=0, test_size=0.25)


def get_performance_report(dataset='credit card fraud', model_nr='rf', model_type='control_fraud', nr_normal_training=[213224],
                           nr_fraud_training=[381], nr_synthetic_fraud_training=[0], test_size=0.25):
    model_dict = {}
    report_dict = {}  # report on recall, precision, etc.
    for i in range(0, len(nr_normal_training)):
        model_name = model_nr +'_'+ model_type +'_'+ str(nr_normal_training[i]) + '_'+str(nr_fraud_training[i]) + '_' + \
                     str(nr_synthetic_fraud_training[i]) +'_ts'+ str(test_size) + '.pkl'
        path = '4) performance improvement/' + dataset + '/control/models/' + model_name

        with open(path, 'rb') as file:
            model = pickle.load(file)
        model_dict[model_name] = model

        model_predictions = model_dict[model_name].predict(X_test)
        model_predictions = [int(round(x)) for x in model_predictions]  # round to be 0 or 1

        report = classification_report(y_test, model_predictions, labels=[0,1],
                                       target_names=['normal', 'fraud'], digits=2, output_dict=True)
        report_dict[model_name] = report

        cm_analysis(y_test, model_predictions, filename='4) performance improvement/' + dataset + '/control/figures/confusion matrices/cm_' + model_name + '.png',
                    labels=[0, 1], ymap=['normal', 'fraud'], title='Model trained on:\n#normal:'+str(nr_normal_training[i])+' #fraud:'+str(nr_fraud_training[i])+'\n'+'  #synthetic fraud:'+str(nr_synthetic_fraud_training[i]))
        plt.close()
    return report_dict




############################# get adding fraud to real data plots ########################



real_and_syn_report = get_performance_report(dataset='credit card fraud', model_nr='rf', model_type='control_fraud',
                                             nr_normal_training=[5381, 5381, 5381, 5381, 5381, 5381, 5381],
                                             nr_fraud_training=[381, 381, 381, 381, 381, 381, 381],
                                             nr_synthetic_fraud_training=[0, 500, 1000, 2000, 3000, 4000, 5000])


# plot real and sythetic fraud only plot
def plot_performance3(x_axis_steps=[0, 500, 1000, 2000, 3000, 4000, 5000], report_dict=real_and_syn_report, fraud_normal='fraud', b='normal',
                      parameter='recall', model='GAN'):

    y_values = list()
    keys = list(report_dict.keys())
    for i in range(0, len(keys)):
        y_values.append(report_dict[keys[i]][fraud_normal][parameter])

    for x, y in zip(x_axis_steps, y_values):
        label = "{:.2f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')

    plt.plot(x_axis_steps, y_values, marker='o', label=parameter+' for fraud')


    y_values = list()
    keys = list(report_dict.keys())
    for i in range(0, len(keys)):
        y_values.append(report_dict[keys[i]][b][parameter])

    for x, y in zip(x_axis_steps, y_values):
        label = "{:.2f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')

    plt.plot(x_axis_steps, y_values, marker='o', label=parameter+' for normal')


    plt.ylabel(parameter)
    plt.xlabel('# synthetic fraud data')
    plt.title('The effect of adding '+model+' synthetic training data on '+parameter)
    plt.grid()
    plt.legend()
    plt.savefig('4) final figures/performance improvement/models/control duplicate fraud/'+parameter+'_'+keys[len(keys)-1]+'.png')
    plt.close()


duplicate_report = get_performance_report(dataset='control duplicate fraud', model_nr='rf', model_type='real_duplicated',
                                          nr_normal_training=[5381, 5381, 5381, 5381, 5381, 5381, 5381],
                                          nr_fraud_training=[381, 381, 381, 381, 381, 381, 381],
                                          nr_synthetic_fraud_training=[0, 500, 1000, 2000, 3000, 4000, 5000])

plot_performance3(x_axis_steps=[0, 500, 1000, 2000, 3000, 4000, 5000], report_dict=duplicate_report, fraud_normal='fraud', b='normal',
                      parameter='f1-score', model='duplicate_fraud')