import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

f, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True, constrained_layout=True)
plt.suptitle('Propensity score results for various datasets', fontsize=14)

# credit card fraud

score_synthpop = 0.19
score_GAN = 0.245
score_cGAN = 0.235
score_WGAN = 0.19
score_WcGAN = 0.17
score_tGAN = 0.23
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

score_synthpop = 0.1498044487129288
score_GAN = 0.24983120846141302
score_cGAN = 0.24999996990374407
score_WGAN = 0.19193968456881755
score_WcGAN = 0.24982153619307615
score_tGAN = 0.18135093659057797
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

score_synthpop = 0.17208491030541753
score_GAN = 0.24996948323493387
score_cGAN =0.249997663829577
score_WGAN =0.24998105907491794
score_WcGAN =0.24999997979218513
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
plt.savefig('3) data quality evaluation/propensity score/propensity_score_all.png')

