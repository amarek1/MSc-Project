# https://nbviewer.jupyter.org/github/codyznash/GANs_for_Credit_Card_Data/blob/master/GAN_comparisons.ipynb#Fig6

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

suffix = '_step_6000'

GAN_dir = '2) synthetic data generation/GAN/credit card fraud/GAN training/'
GAN_losses = pickle.load(open(GAN_dir + 'adam_GAN_losses'+suffix+'.pkl','rb'))

CGAN_dir = '2) synthetic data generation/CGAN/credit card fraud/CGAN training/'
CGAN_losses = pickle.load(open(CGAN_dir + 'adam_CGAN_losses'+suffix+'.pkl','rb'))

WGAN_dir = '2) synthetic data generation/WGAN/credit card fraud/WGAN training/'
WGAN_losses = pickle.load(open(WGAN_dir + 'adam_WGAN_losses'+suffix+'.pkl','rb'))

WCGAN_dir = '2) synthetic data generation/WcGAN/credit card fraud/WcGAN training/'
WCGAN_losses = pickle.load(open(WCGAN_dir + 'Adam_l2_WCGAN_losses'+suffix+'.pkl','rb'))

# Look at the unsmoothed losses

data_fields = ['combined_losses_', 'real_losses_', 'generated_losses_', 'xgb_losses']
sampling_intervals = [ 1, 1, 1, 1 ]
labels = [ 'GAN','CGAN','WGAN','WCGAN' ]
linestyles = ['-', '--', '-.', ':']

# for data_ix in range(len(data_fields)):
#     data_sets = [ GAN_losses[data_ix], CGAN_losses[data_ix], WGAN_losses[data_ix], WCGAN_losses[data_ix]]
#
#     plt.figure(figsize=(10,5))
#     for data, label, linestyle in zip(data_sets, labels, linestyles):
#         plt.plot( np.array(range(0,len(data)))*sampling_intervals[data_ix],
#                  data,
#                  label=label, linestyle=linestyle )
#
#     plt.ylabel(data_fields[data_ix])
#     plt.xlabel('training step')
#     plt.legend()
#     # plt.show()


# look at smoothed losses
data_fields = ['combined_losses_', 'real_losses_', 'generated_losses_', 'xgb_losses']
sampling_intervals = [ 1, 1, 1, 10 ]
labels = [ 'GAN','CGAN','WGAN','WCGAN' ]
linestyles = ['-', '--', '-.', ':']
w = 100

data_ix = 3
data_sets = [ GAN_losses[data_ix], CGAN_losses[data_ix], WGAN_losses[data_ix], WCGAN_losses[data_ix]]

plt.figure(figsize=(10,5))
for data, label, linestyle in zip(data_sets, labels, linestyles):
    plt.plot( np.array(range(0,len(data)))*sampling_intervals[data_ix],
             pd.DataFrame(data).rolling(w).mean(),
             label=label, linestyle=linestyle )

plt.ylabel(data_fields[data_ix])
plt.xlabel('training step')
plt.grid()
plt.legend()
plt.title('xgboost performance on generated data detection (smoothed)')
plt.savefig('4) final figures/best GAN assessment/xgboost_performance_smoothed.png')
