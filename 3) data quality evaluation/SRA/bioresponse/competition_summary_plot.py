import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats
np.random.seed(7)

score_synthpop = 0.4
score_GAN = 0
score_cGAN = 0
score_WGAN = 0
score_WcGAN =0
score_tGAN = 0


fig = plt.figure()
x = np.arange(6)
distance = [score_synthpop, score_GAN, score_cGAN, score_WGAN, score_WcGAN, score_tGAN]
plt.title('Synthetic ranking agreement results for bioresponse dataset')
plt.ylabel('Score')
plt.xlabel('data generator')
a = plt.bar(x, distance)
plt.xticks(x, ('synthpop', 'GAN', 'cGAN', 'WGAN', 'WcGAN', 'tGAN'))
plt.tight_layout()
plt.ylim((0,1))
a[0].set_color('yellowgreen')#('lightseagreen')
a[1].set_color('gold')
a[2].set_color('darkorange')
a[3].set_color('grey')
a[4].set_color('royalblue')#('dodgerblue')
a[5].set_color('mediumvioletred')
plt.savefig('3) data quality evaluation/competition measure/bioresponse/competition_measure_plot.png')