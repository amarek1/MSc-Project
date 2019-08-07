import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats
np.random.seed(7)

score_synthpop = 1
score_GAN = 0.8
score_cGAN = 0.9
score_WGAN = 1
score_WcGAN = 1
score_tGAN = 0.3


fig = plt.figure()
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]
plt.title('Synthetic ranking agreement results for fraud dataset')
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
plt.savefig('3) data quality evaluation/competition measure/credit card fraud/competition_measure_plot.png')