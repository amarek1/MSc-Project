import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

score_synthpop = 0.0875369132265257
score_GAN = 0.7243958554098529
score_cGAN = 0.23846898572854103
score_WGAN = 0.8396380451713511
score_WcGAN = 0.18841910843517576
score_tGAN = 0


fig = plt.figure()
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]
plt.title('Correlation mean difference for bioresponse dataset')
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
plt.savefig('3) data quality evaluation/qualitative distribution assesment/bioresponse/correlation plots/correlation_summary.png')