import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

f, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True, constrained_layout=True)
plt.suptitle('Correlation mean difference for various datasets', fontsize=14)

# credit card fraud

score_synthpop = 0.0486
score_GAN = 0.3
score_cGAN = 0.25
score_WGAN = 0.062
score_WcGAN = 0.0551
score_tGAN = 0.101
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]

ax[0].set_title('Credit card fraud',fontsize=13)
ax[0].set_ylabel('Score')
ax[0].set_xlabel('data generator')
a = ax[0].bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))

a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')


# customer churn

score_synthpop = 0.02127
score_GAN = 0.25511
score_cGAN = 0.4
score_WGAN = 0.02423
score_WcGAN = 0.24698
score_tGAN = 0.0458
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]

ax[1].set_title('Customer churn',fontsize=13)
ax[1].set_ylabel('Score')
ax[1].set_xlabel('data generator')
a = ax[1].bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))

a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')




# bioresponse

score_synthpop = 0.0875369132265257
score_GAN = 0.7243958554098529
score_cGAN = 0.23846898572854103
score_WGAN = 0.8396380451713511
score_WcGAN = 0.18841910843517576
score_tGAN = 0
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]

ax[2].set_title('Bioresponse',fontsize=13)
ax[2].set_ylabel('Score')
ax[2].set_xlabel('data generator')
a = ax[2].bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))

a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')
plt.savefig('3) data quality evaluation/qualitative distribution assesment/correlation_all.png')

