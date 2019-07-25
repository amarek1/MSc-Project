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

# load synthetic fraud examples
file_name = '2) synthetic data generation/WcGAN/credit card fraud/WcGAN results/WcGAN_fraud_5904_Adam.pkl'
syn_fraud = pd.read_pickle(file_name)


# # load the synthetic data
# file_name = 'C:/Users/amarek/Desktop/R/synthpop/syntpop_data_cart_fo.csv'
# syn_data = pd.read_csv(file_name)

# spec = importlib.util.spec_from_file_location("TGAN.py",
#                                               "C:/Users/amarek/PycharmProjects/data_lab/"
#                                               "data generation/GANs/TGAN.py")
# A = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(A)
# only extract the fraud transactions
# syn_fraud = syn_data.loc[syn_data.loc[:, 'class'] == 1, :]

real_fraud = real_data.loc[real_data.loc[:, 'class'] == 1, :]
syn_fraud = syn_fraud[:len(real_fraud)]
# syn_fraud = A.df
# syn_fraud['class'] = ori_fraud['class'].values
n_features = len(real_data.columns)
cols = list(range(0, n_features))


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

# get_ks_test_results(ori_data,syn_data)

# function to plot correlation matrix
def correlation_matrix(df, title, file=""):
    plt.figure(figsize=(15, 15))
    corr_plot = sns.heatmap(df, annot=False, square=True, cmap='coolwarm', vmax=0.5, vmin=-0.5).set_title(title)

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
    rescaled_data['class'] = data_array.loc[:, ['class']].values  # keep class the same as original
    return rescaled_data


# function comparing synthetic data with the original
# generator and path are strings
def compare_data(original_data, synthetic_data, datatype, generator):

    f, axes = plt.subplots(n_features, figsize=(9, 100))
    axes[0].set_title(datatype+'Original and syn data generated using'+generator, fontsize='x-large')
    plt.tight_layout()
    pearsonrs = []
    intersections = []
    method = []

    for i in cols:
        s = sns.distplot(original_data.iloc[:, i], hist=False, rug=False, label="Synthetic-" + str(i), ax=axes[i],
                         kde_kws={'linestyle': '-', 'linewidth': 4})

        sns.distplot(synthetic_data.iloc[:, i], hist=False, rug=False, label="Original-" + str(i), ax=axes[i],
                     kde_kws={'linestyle': '--', 'linewidth': 2})

        o_hist, _ = np.histogram(original_data.iloc[:, i], bins=100)
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

    f.savefig('3) data quality evaluation/qualitative distribution assesment/credit card fraud/histograms/'+datatype + '_histograms_' + generator + '.png', bbox_inches='tight')

    original_corr = original_data.iloc[:, :-1].corr()

    synthetic_corr = synthetic_data.iloc[:, :-1].corr()

    abs_mean_corr = np.mean(np.mean(abs(original_corr - synthetic_corr)))

    print("\nCorrelation Matrix - {}".format(generator))
    correlation_matrix(synthetic_corr, datatype + 'Generated Data ' + generator)

    print("Correlation Matrix Diff - {}".format(generator))
    correlation_matrix(original_corr - synthetic_corr, datatype + 'Correlation Difference ' + generator,
                       '3) data quality evaluation/qualitative distribution assesment/credit card fraud/correlation plots/'+datatype + '_correlation_' + generator)

    summary_df = pd.DataFrame(data={'method': method,
                                    'feature': cols,
                                    'pearsonr': pearsonrs,
                                    'intersection': intersections,
                                    'abs_mean_corr': abs_mean_corr})

    summary_df.to_csv('3) data quality evaluation/qualitative distribution assesment/credit card fraud/'+datatype + 'summary_stats_' + generator, index=False)


def get_scatters(original_data, synthetic_data, datatype, generator, feature_position):

    names = original_data.columns
    fig, axs = plt.subplots(len(cols), 2, figsize=(9, 110))  # sharex=True, sharey=True)
    fig.suptitle(datatype + 'scatter plots for the feature '+names[feature_position] + '_' + generator,
                 fontsize='x-large')

    for i in range(0, len(cols)):
        axs[i, 0].title.set_text('original')
        axs[i, 0].set_xlabel(names[i])
        axs[i, 0].set_ylabel(names[feature_position])
        axs[i, 0].scatter(y=original_data.iloc[:, feature_position], x=original_data.iloc[:, i], c='blue', alpha=0.4)
        axs[i, 0].grid()

        axs[i, 1].title.set_text('synthetic')
        axs[i, 1].set_xlabel(names[i])
        axs[i, 1].set_ylabel(names[feature_position])
        axs[i, 1].scatter(y=synthetic_data.iloc[:, feature_position], x=synthetic_data.iloc[:, i], c='blue', alpha=0.4)
        axs[i, 1].grid()

    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig('3) data quality evaluation/qualitative distribution assesment/credit card fraud/scatter plots/'+datatype + '_scatter_plots_'+names[feature_position] + '_' + generator + '.png')
    plt.close(fig)


print(real_fraud, syn_fraud)
# compare_data(ori_data, syn_data, 'f+nf', 'synthpop_cart')
ori_fraud_i = real_fraud + 0.000001  # correct singular matrix error
syn_fraud_i = syn_fraud + 0.000001
compare_data(ori_fraud_i, syn_fraud_i, 'fraud', 'WcGAN_fraud_5904_Adam')



# ori_data = rescale_0_1(real_data)
# syn_data = rescale_0_1(syn_data)
real_fraud = rescale_0_1(real_fraud)
syn_fraud = rescale_0_1(syn_fraud)


# for i in range(0, 31):
#     get_scatters(ori_data, syn_data, 'f+nf', 'synthpop_lin_reg', i)


for i in range(0, 31):
    get_scatters(real_fraud, syn_fraud, 'fraud', 'WcGAN_fraud_5904_Adam', i)
