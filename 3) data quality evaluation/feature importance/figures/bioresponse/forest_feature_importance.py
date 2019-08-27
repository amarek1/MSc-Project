# idea from Privacy-preserving generative deep neural networks support clinical data sharing
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats
np.random.seed(7)


def unpack_model(model_name):
    path = '3) data quality evaluation/feature importance/models/bioresponse/' + model_name
    with open(path, 'rb') as file:
        model = pickle.load(file)
        return model

real_model = unpack_model('FI_real_2034_350_0_ts0.25.pkl')
synthpop_model = unpack_model('FI_synthpop_2034_0_350_ts0.25.pkl')
GAN_model = unpack_model('FI_GAN_2034_0_350_ts0.25.pkl')
cGAN_model = unpack_model('FI_cGAN_2034_0_350_ts0.25.pkl')
WGAN_model = unpack_model('FI_WGAN_2034_0_350_ts0.25.pkl')
WcGAN_model = unpack_model('FI_WcGAN_2034_0_350_ts0.25.pkl')
tGAN_model = unpack_model('FI_WcGAN_2034_0_350_ts0.25.pkl')



# load the original data
file_name = 'data/bioresponse/bio_clean.pkl'
real_data = pd.read_pickle(file_name)

# unbalanced data
X = real_data.drop('class', axis=1)
y = real_data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


def get_feature_importance(model, x_train):
    feature_importance = pd.DataFrame(model.feature_importances_, index=x_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)
    return feature_importance


FI_real = get_feature_importance(real_model, X_train)
FI_synthpop = get_feature_importance(synthpop_model, X_train)
FI_GAN = get_feature_importance(GAN_model, X_train)
FI_cGAN = get_feature_importance(cGAN_model, X_train)
FI_WGAN = get_feature_importance(WGAN_model, X_train)
FI_WcGAN = get_feature_importance(WcGAN_model, X_train)
FI_tGAN = get_feature_importance(tGAN_model, X_train)

row_names_real = list(FI_real.index)
row_names_synthpop = list(FI_synthpop.index)
row_names_GAN = list(FI_GAN.index)
row_names_cGAN = list(FI_cGAN.index)
row_names_WGAN = list(FI_WGAN.index)
row_names_WcGAN = list(FI_WcGAN.index)
row_names_tGAN = list(FI_tGAN.index)

print(row_names_real)
plt.scatter(row_names_real, FI_real.loc[:, 'importance'], s=40, c='r', marker='x', label='real')
plt.scatter(row_names_synthpop, FI_synthpop.loc[:, 'importance'], s=25, c='lightseagreen', marker='o', label='synthpop')
plt.scatter(row_names_GAN, FI_GAN.loc[:, 'importance'], s=25, c='gold', marker='o', label='GAN')
plt.scatter(row_names_cGAN, FI_cGAN.loc[:, 'importance'], s=25, c='darkorange', marker='o', label='cGAN')
plt.scatter(row_names_WGAN, FI_WGAN.loc[:, 'importance'], s=25, c='grey', marker='o', label='WGAN')
plt.scatter(row_names_tGAN, FI_tGAN.loc[:, 'importance'], s=25, c='dodgerblue', marker='o', label='tGAN')
plt.scatter(row_names_WcGAN, FI_WcGAN.loc[:, 'importance'], s=25, c='mediumvioletred', marker='o', label='WcGAN')
plt.show()


plt.xlabel('features', fontsize=10)
plt.ylabel('importance score (x$10^-2$)', fontsize=10)
plt.legend(loc='upper right')
plt.xticks(fontsize=8, rotation=80)
plt.title('Random forest feature importance scores\nfor bioresponse dataset')
plt.grid()
plt.tight_layout()
plt.savefig('3) data quality evaluation/feature importance/figures/bioresponse/FI_scatterplot.png')


# calculate overall score

FI_real = FI_real+0.001
dist_synthpop = float(sum(abs(FI_real.values-FI_synthpop.values)/FI_real.values)/len(FI_real.values))
dist_GAN = float(sum(abs(FI_real.values-FI_GAN.values)/FI_real.values)/len(FI_real.values))
dist_cGAN = float(sum(abs(FI_real.values-FI_cGAN.values)/FI_real.values)/len(FI_real.values))
dist_WGAN = float(sum(abs(FI_real.values-FI_WGAN.values)/FI_real.values)/len(FI_real.values))
dist_WcGAN = float(sum(abs(FI_real.values-FI_WcGAN.values)/FI_real.values)/len(FI_real.values))
dist_tGAN = float(sum(abs(FI_real.values-FI_tGAN.values)/FI_real.values)/len(FI_real.values))

# dist_synthpop = float(sum(abs(FI_real.values-FI_synthpop.values))/len(FI_real.values))
# dist_GAN = float(sum(abs(FI_real.values-FI_GAN.values))/len(FI_real.values))
# dist_cGAN = float(sum(abs(FI_real.values-FI_cGAN.values))/len(FI_real.values))
# dist_WGAN = float(sum(abs(FI_real.values-FI_WGAN.values))/len(FI_real.values))
# dist_WcGAN = float(sum(abs(FI_real.values-FI_WcGAN.values))/len(FI_real.values))
# dist_tGAN = float(sum(abs(FI_real.values-FI_tGAN.values))/len(FI_real.values))


fig = plt.figure()
x = np.arange(6)
distance = [dist_synthpop, dist_GAN, dist_cGAN, dist_WGAN, dist_WcGAN, 0]
plt.title('Distance between real and synthetic feature importance scores\nfor bioresponse dataset')
plt.ylabel('normalised distance')
plt.xlabel('data generator')
a = plt.bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))
plt.tight_layout()
a[0].set_color('yellowgreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')
a[5].set_color('mediumvioletred')
plt.savefig('3) data quality evaluation/feature importance/figures/bioresponse/FI_score_normalised.png')
