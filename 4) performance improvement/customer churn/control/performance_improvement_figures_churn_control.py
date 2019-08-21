import pandas as pd
import numpy as np
from global_functions import cm_analysis
import pickle
from global_functions import get_data
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
np.random.seed(7)

# load the data
file_name = 'data/customer churn/customer churn modified.pkl'
data = pd.read_pickle(file_name)

# get same train:test split as when training the model
# even out the data set -> 1:1 ratio of fraud and non fraud
X_train, X_test, y_train, y_test = get_data(real_data=data, synthetic_data=data,
                    nr_normal_training=381, nr_fraud_training=381, nr_synthetic_fraud_training=0, test_size=0.25)



############################# open models and get classification reports ########################


def get_performance_report(dataset='customer churn', model_nr='rf', model_type='control_churn', nr_normal_training=[213224],
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

        # plot confusion matrix
        cm_analysis(y_test, model_predictions, filename='4) performance improvement/' + dataset + '/control/figures/confusion matrices/cm_' + model_name + '.png',
                    labels=[0, 1], ymap=['normal', 'fraud'], title='RF model trained on\n#normal:'+str(nr_normal_training[i])+' #churn:'+str(nr_fraud_training[i])+'\n'+'  #duplicated churn:'+str(nr_synthetic_fraud_training[i]))
        plt.close()
    return report_dict




############################# extract performance parameters and plot########################



real_and_syn_report = get_performance_report(dataset='customer churn', model_nr='rf', model_type='control_churn',
                                             nr_normal_training=[3847, 3847, 3847, 3847, 3847, 3847],
                                             nr_fraud_training=[1435, 1435, 1435, 1435, 1435, 1435],
                                             nr_synthetic_fraud_training=[0, 500, 1000, 1500, 2000, 2500])


# plot real and sythetic fraud only plot
def plot_performance3(x_axis_steps=[0, 500, 1000, 1500, 2000, 2500], report_dict=real_and_syn_report, fraud_par='fraud', normal_par='normal',
                      parameter='recall', model='GAN'):

    y_values = list()
    keys = list(report_dict.keys())
    for i in range(0, len(keys)):
        y_values.append(report_dict[keys[i]][fraud_par][parameter])

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
        y_values.append(report_dict[keys[i]][normal_par][parameter])

    for x, y in zip(x_axis_steps, y_values):
        label = "{:.2f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')

    plt.plot(x_axis_steps, y_values, marker='o', label=parameter+' for normal')


    plt.ylabel(parameter)
    plt.xlabel('# duplicated churn data')
    plt.title('The effect of adding duplicated churn data on '+parameter)
    plt.grid()
    plt.legend()
    plt.savefig('4) performance improvement/customer churn/control/figures/plots/'+parameter+'_'+keys[len(keys)-1]+'.png')
    plt.close()

###################################### run the functions ###################################

control_report = get_performance_report(dataset='customer churn', model_nr='rf', model_type='control_churn',
                                        nr_normal_training=[3847, 3847, 3847, 3847, 3847, 3847],
                                        nr_fraud_training=[1435, 1435, 1435, 1435, 1435, 1435],
                                        nr_synthetic_fraud_training=[0, 500, 1000, 1500, 2000, 2500])

plot_performance3(x_axis_steps=[0, 500, 1000, 1500, 2000, 2500], report_dict=control_report, fraud_par='fraud', normal_par='normal',
                  parameter='f1-score', model='control_fraud')

control_report = get_performance_report(dataset='customer churn', model_nr='rf', model_type='control_churn',
                                        nr_normal_training=[3847, 3847, 3847, 3847, 3847, 3847],
                                        nr_fraud_training=[1435, 1435, 1435, 1435, 1435, 1435],
                                        nr_synthetic_fraud_training=[0, 500, 1000, 1500, 2000, 2500])

plot_performance3(x_axis_steps=[0, 500, 1000, 1500, 2000, 2500], report_dict=control_report, fraud_par='fraud', normal_par='normal',
                  parameter='precision', model='control_fraud')

control_report = get_performance_report(dataset='customer churn', model_nr='rf', model_type='control_churn',
                                        nr_normal_training=[3847, 3847, 3847, 3847, 3847, 3847],
                                        nr_fraud_training=[1435, 1435, 1435, 1435, 1435, 1435],
                                        nr_synthetic_fraud_training=[0, 500, 1000, 1500, 2000, 2500])

plot_performance3(x_axis_steps=[0, 500, 1000, 1500, 2000, 2500], report_dict=control_report, fraud_par='fraud', normal_par='normal',
                  parameter='recall', model='control_fraud')


