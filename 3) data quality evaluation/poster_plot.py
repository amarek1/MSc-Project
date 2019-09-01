import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

f, ax = plt.subplots(2,2, figsize=(8, 8), sharey=False, sharex=True, constrained_layout=True)
plt.suptitle('Synthetic data quality evaluation', fontsize=14)

# SRA

score_synthpop = 0.0486
score_GAN = 0.3
score_cGAN = 0.25
score_WGAN = 0.062
score_WcGAN = 0.0551
score_tGAN = 0.101
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]

ax[0,0].set_title('Synthetic ranking assessment',fontsize=13)
ax[0,0].set_ylabel('Score')
ax[0,0].set_xlabel('data generator')
a = ax[0,0].bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))

a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')


# propensity score

score_synthpop = 0.19
score_GAN = 0.245
score_cGAN = 0.235
score_WGAN = 0.19
score_WcGAN = 0.17
score_tGAN = 0.23
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]

ax[0,1].set_title('Propensity score',fontsize=13)
ax[0,1].set_ylabel('Score')
ax[0,1].set_xlabel('data generator')
a = ax[0,1].bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))

a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')

# correlation

score_synthpop = 0.0486
score_GAN = 0.3
score_cGAN = 0.25
score_WGAN = 0.062
score_WcGAN = 0.0551
score_tGAN = 0.101
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]

ax[1,0].set_title('Correlation mean difference',fontsize=13)
ax[1,0].set_ylabel('Score')
ax[1,0].set_xlabel('data generator')
a = ax[1,0].bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))

a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')

# FI

score_synthpop = 0.30397905792328034
score_GAN = 0.6095067062886259
score_cGAN = 0.5043027948775269
score_WGAN = 0.49373622393187355
score_WcGAN = 0.38102711907990955
score_tGAN = 0.45461186162065087
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]

ax[1,1].set_title('Feature importance mean distance',fontsize=13)
ax[1,1].set_ylabel('Score')
ax[1,1].set_xlabel('data generator')
a = ax[1,1].bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))

a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')

plt.savefig('3) data quality evaluation/poster_all.png')