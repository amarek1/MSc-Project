# import importlib.util as import_functions
# spec = import_functions.spec_from_file_location('global_functions.py',
#                                               'C:/Users/amarek/PycharmProjects/data_lab_clean/global_functions.py')
# catch = import_functions.module_from_spec(spec)
# spec.loader.exec_module(catch)

import pandas as pd
from sklearn.model_selection import train_test_split

# create train and test data with equal number of classes
def get_balanced_data(data):

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

# print out confusion matrix and values like precision, recall, f1-score
def get_model_performance(model, model_name, x_test, y_test):
    from sklearn.metrics import confusion_matrix, classification_report
    predictions = model.predict(x_test)
    r_predictions = [int(round(x)) for x in predictions]
    cm = confusion_matrix(y_test, r_predictions)
    print('=============================================')
    print('\n'+model_name+'\n')
    print('---------------------------------------------')
    print('confusion matrix\n')
    print(cm,'\n')
    print('---------------------------------------------')
    print(classification_report(y_test, r_predictions))
    print('=============================================')
    return r_predictions


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.utils.multiclass import unique_labels
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = list(unique_labels(y_true, y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(3.5,3.75), title='Confusion matrix'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            s = cm_sum[i]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, loc='center', pad=15, fontsize=10)
    sns.heatmap(cm_perc, annot=annot, fmt='', ax=ax, cmap='Blues', xticklabels=ymap, yticklabels=ymap, cbar=False)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    fig.tight_layout()
    plt.savefig(filename)


def get_data(real_data, synthetic_data, nr_normal_training, nr_fraud_training, nr_synthetic_fraud_training,
             test_size):

    train_data, test_data, train_labels, test_labels = \
        train_test_split(real_data, real_data['class'], test_size=test_size, random_state=1)

    # even out the data set -> 1:1 ratio of 0 and 1 classes
    data_training = train_data.sample(frac=1)  # shuffle
    data_testing = test_data.sample(frac=1)  # shuffle

    fraud_data_training = data_training.loc[data_training['class'] == 1][:nr_fraud_training]
    fraud_data_testing = data_testing.loc[data_testing['class'] == 1]

    non_fraud_data_training = data_training.loc[data_training['class'] == 0][:nr_normal_training]
    non_fraud_data_testing = data_testing.loc[data_testing['class'] == 0][:len(fraud_data_testing)]

    synthetic_data = synthetic_data.sample(frac=1)
    synthetic_fraud = synthetic_data.loc[synthetic_data['class'] == 1][:nr_synthetic_fraud_training]

    fraud_data_training = pd.concat([fraud_data_training, synthetic_fraud])

    even_data_training = pd.concat([fraud_data_training, non_fraud_data_training])
    even_data_testing = pd.concat([fraud_data_testing, non_fraud_data_testing])

    even_data_training = even_data_training.sample(frac=1, random_state=42)
    even_data_testing = even_data_testing.sample(frac=1, random_state=42)

    train_data = even_data_training.drop('class', axis=1)
    test_data = even_data_testing.drop('class', axis=1)
    train_labels = even_data_training['class']
    test_labels = even_data_testing['class']

    return train_data, test_data, train_labels, test_labels