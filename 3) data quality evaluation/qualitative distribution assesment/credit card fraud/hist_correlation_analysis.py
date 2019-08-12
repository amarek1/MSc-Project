import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import importlib.util
np.random.seed(1)

###################### load the data ####################################

# load the real data
file_name = 'data/credit card fraud/data_creditcard.pkl'  # set working directory to MSc Project
real_data = pd.read_pickle(file_name)

# # load synthetic fraud examples
# file_name = '2) synthetic data generation/WcGAN/credit card fraud/WcGAN results/WcGAN_fraud_492_Adam_l1_.pkl'
# syn_fraud = pd.read_pickle(file_name)

# # load the synthetic data
# file_name = 'C:/Users/amarek/Desktop/R/synthpop/syntpop_data_cart_fo.csv'
# syn_fraud = pd.read_csv(file_name)

# load synthetic fraud examples
file_name = '2) synthetic data generation/tGAN/customer churn/churn/tGAN_churn_50002.pkl'
syn_fraud = pd.read_pickle(file_name)


real_fraud = real_data.loc[real_data.loc[:, 'class'] == 1, :]
syn_fraud = syn_fraud[:len(real_fraud)]

n_features = len(real_data.columns)
cols = list(range(0, n_features))


########################## KS-test ######################################
# function returning p-values as a result of KS-test
# it also returns true and false, true meaning that we fail to reject
# the hypothesis that the two samples are the same
def get_ks_test_results(data1, data2):
    ks_test_results = []
    column_names = data1.columns
    for i in range(0, len(data1.columns)):
        a = stats.ks_2samp(data1[column_names[i]], data2[column_names[i]])
        ks_test_results.append(list(a))
    cols = data1.columns
    dictionary = dict(zip(cols, ks_test_results))
    pvalue_table = pd.DataFrame.from_dict(dictionary)
    pvalue_table.rename(index={0: 'K-S statistic', 1: 'p-value'}, inplace=True)
    return(print('\np-values\n\n', pvalue_table.loc['p-value', :],
                 '\n\n\nif True - data "the same"\n\n', pvalue_table.loc['p-value', :] >= 0.05))

# If the K-S statistic is small or the p-value is high,
# then we cannot reject the hypothesis that the distributions
# of the two samples are the same

# get_ks_test_results(real_data,syn_data)



############################### correlation matrix and histograms #######################################

# function to plot correlation matrix
def correlation_matrix(df, title, file=""):
    plt.figure(figsize=(15, 15))
    corr_plot = sns.heatmap(df, annot=False, square=True, cmap='coolwarm', vmax=0.5, vmin=-0.5).set_title(title, fontsize=20)

    if file != "":
        fig = corr_plot.get_figure()
        fig.savefig(file + '.png', bbox_inches='tight')


# function to calculate histogram intersection
def return_intersection(hist1, hist2):
    minima = np.minimum(hist1, hist2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist2))
    return intersection


def rescale_0_1(data_array):
    rescaled_data = pd.DataFrame()
    names = data_array.columns
    for i in range(0, len(names)):
        a = np.interp(data_array[names[i]], (data_array[names[i]].min(), data_array[names[i]].max()), (0, 1))
        rescaled_data[names[i]] = a
    rescaled_data['class'] = data_array.loc[:, ['class']].values  # keep class the same as real
    return rescaled_data


# function comparing synthetic data with the real
# generator and path are strings
def compare_data(real_data, synthetic_data, datatype, generator):

    f, axes = plt.subplots(n_features, figsize=(9, 100))
    axes[0].set_title('Real and synthetic data histograms for '+generator, fontsize='x-large')
    plt.tight_layout()
    pearsonrs = []
    intersections = []
    method = []

    for i in cols:
        s = sns.distplot(real_data.iloc[:, i], hist=False, rug=False, label="Synthetic-" + str(i), ax=axes[i],
                         kde_kws={'linestyle': '-', 'linewidth': 4})

        sns.distplot(synthetic_data.iloc[:, i], hist=False, rug=False, label="Real-" + str(i), ax=axes[i],
                     kde_kws={'linestyle': '--', 'linewidth': 2})

        o_hist, _ = np.histogram(real_data.iloc[:, i], bins=100)
        g_hist, _ = np.histogram(synthetic_data.iloc[:, i], bins=100)
        intersection = return_intersection(o_hist, g_hist)
        pearson = stats.pearsonr(o_hist, g_hist)

        pearsonrs.append(pearson[0])
        intersections.append(intersection)
        method.append(generator)

        print("Intersection for {}: {:03.2f} ".format(i, intersection))
        # axes[i].text(0.15, 0.85,'Intersection:{} '.format(intersection), fontsize=9) #add text
        s.text(0.02, 0.90, 'Pearson Coeff:',
               fontsize=8, horizontalalignment='left',
               verticalalignment='center',
               transform=axes[i].transAxes)
        s.text(0.2, 0.90, '{:05.3f}'.format(pearson[0]),
               fontsize=8, horizontalalignment='left',
               verticalalignment='center',
               transform=axes[i].transAxes)

        s.text(0.02, 0.80, 'Intersection:',
               fontsize=8, horizontalalignment='left',
               verticalalignment='center',
               transform=axes[i].transAxes)
        s.text(0.2, 0.80, '{:05.3f}'.format(intersection),
               fontsize=8, horizontalalignment='left',
               verticalalignment='center',
               transform=axes[i].transAxes)

    f.savefig('3) data quality evaluation/qualitative distribution assesment/credit card fraud/histograms/histograms_' + generator + '.png', bbox_inches='tight')


    real_corr = real_data.iloc[:, :-1].corr()
    synthetic_corr = synthetic_data.iloc[:, :-1].corr()
    abs_mean_corr = np.mean(np.mean(abs(real_corr - synthetic_corr)))


    correlation_matrix(synthetic_corr, datatype + 'Generated Data ' + generator)

    correlation_matrix(real_corr - synthetic_corr, 'Correlation Difference for ' + generator+'\n Absolute mean correlation difference: '+ str(abs_mean_corr),
                       '3) data quality evaluation/qualitative distribution assesment/credit card fraud/correlation plots/correlation_' + generator)

    summary_df = pd.DataFrame(data={'method': method,
                                    'feature': cols,
                                    'pearsonr': pearsonrs,
                                    'intersection': intersections,
                                    'abs_mean_corr': abs_mean_corr})

    summary_df.to_csv('3) data quality evaluation/qualitative distribution assesment/credit card fraud/stats/summary_stats_' + generator, index=False)


####################################### scatter plots #####################################################

def get_scatters(real_data, synthetic_data, datatype, generator, feature_position):

    names = real_data.columns
    fig, axs = plt.subplots(len(cols), 2, figsize=(9, 110))  # sharex=True, sharey=True)
    fig.suptitle('Scatter plots for the feature '+names[feature_position] + ' and generator ' + generator,
                 fontsize='x-large')

    for i in range(0, len(cols)):
        axs[i, 0].title.set_text('real')
        axs[i, 0].set_xlabel(names[i])
        axs[i, 0].set_ylabel(names[feature_position])
        axs[i, 0].scatter(y=real_data.iloc[:, feature_position], x=real_data.iloc[:, i], c='blue', alpha=0.4)
        axs[i, 0].grid()

        axs[i, 1].title.set_text('synthetic')
        axs[i, 1].set_xlabel(names[i])
        axs[i, 1].set_ylabel(names[feature_position])
        axs[i, 1].scatter(y=synthetic_data.iloc[:, feature_position], x=synthetic_data.iloc[:, i], c='blue', alpha=0.4)
        axs[i, 1].grid()

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig('3) data quality evaluation/qualitative distribution assesment/credit card fraud/scatter plots/scatter_plots_'+names[feature_position] + '_' + generator + '.png')
    plt.close(fig)



##################################### run the functions ###############################################

# compare_data(real_data, syn_data, 'f+nf', 'synthpop_cart')
real_fraud_i = real_fraud + 0.000001  # correct singular matrix error
syn_fraud_i = syn_fraud + 0.000001

compare_data(real_fraud_i, syn_fraud_i, 'fraud', 'tGAN_test')

# real_data = rescale_0_1(real_data)
# syn_data = rescale_0_1(syn_data)

# for i in range(0, 31):
#     get_scatters(real_data, syn_data, 'f+nf', 'synthpop_lin_reg', i)


# real_fraud = rescale_0_1(real_fraud)
# syn_fraud = rescale_0_1(syn_fraud)
#
# for i in range(16, 17):
#     get_scatters(real_fraud, syn_fraud, 'fraud', 'WcGAN_fraud_492_Adam', i)
