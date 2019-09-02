import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

f, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True, constrained_layout=True)
plt.suptitle('Distance between real and synthetic features importance scores for various datasets', fontsize=14)

# credit card fraud

score_synthpop = 0.30397905792328034
score_GAN = 0.6095067062886259
score_cGAN = 0.5043027948775269
score_WGAN = 0.49373622393187355
score_WcGAN = 0.38102711907990955
score_tGAN = 0.45461186162065087
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

score_synthpop = 0.41538877397257806
score_GAN = 0.9051305421140329
score_cGAN = 0.9690531101337705
score_WGAN = 0.6749849384120223
score_WcGAN = 0.6899525790251759
score_tGAN = 0.6692463017386511
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

score_synthpop = 0.8822808778630279
score_GAN = 0.9992266815798903
score_cGAN = 1.0070305989840402
score_WGAN = 0.9961532318751836
score_WcGAN =1.0005456534634933
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
plt.savefig('3) data quality evaluation/feature importance/FI_all.png')

