import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

score_synthpop = 0.0486
score_GAN = 0.3
score_cGAN = 0.25
score_WGAN = 0.062
score_WcGAN = 0.0551
score_tGAN = 0.101


fig = plt.figure()
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]
plt.title('Correlation mean difference')
plt.ylabel('Score')
plt.xlabel('data generator')
a = plt.bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))
plt.tight_layout()
a[0].set_color('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('dodgerblue')
a[5].set_color('mediumvioletred')
plt.savefig('3) data quality evaluation/qualitative distribution assesment/credit card fraud/correlation plots/correlation_summary.png')