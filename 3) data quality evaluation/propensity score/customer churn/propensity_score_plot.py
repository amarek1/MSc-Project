import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

score_synthpop = 0.1498044487129288
score_GAN = 0.24998962490330084
score_cGAN = 0.24999996785364897
score_WGAN = 0.24718770292780728
score_WcGAN = 0.2498946456253423
score_tGAN = 0.23964328460769005



fig = plt.figure()
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]
plt.title('Synthetic ranking agreement results for churn dataset')
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
plt.savefig('3) data quality evaluation/propensity score/customer churn/propensity_score_plot_churn.png')