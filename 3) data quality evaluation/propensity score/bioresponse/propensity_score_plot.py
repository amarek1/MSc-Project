import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

score_synthpop = 0.17208491030541753
score_GAN = 0.24996948323493387
score_cGAN =0.249997663829577
score_WGAN =0.24998105907491794
score_WcGAN =0.24999997979218513
score_tGAN = 0


fig = plt.figure()
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]
plt.title('Propensity score for bioresponse dataset')
plt.ylabel('Score')
plt.xlabel('data generator')
a = plt.bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))
plt.tight_layout()
a[0].set_color('yellowgreen')#'lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#'dodgerblue')
a[5].set_color('mediumvioletred')
plt.savefig('3) data quality evaluation/propensity score/bioresponse/propensity_score_plot.png')