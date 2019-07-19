import pandas as pd
import numpy as np
import importlib.util
import pickle

spec = importlib.util.spec_from_file_location("GAN code.py",
                                              "C:/Users/amarek/PycharmProjects/data_lab/"
                                              "data generation/GANs/GAN code.py")
GAN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GAN)

cluster_spec = importlib.util.spec_from_file_location("k means classes.py",
                                                      "C:/Users/amarek/PycharmProjects/data_lab/"
                                                      "data generation/GANs/k means classes.py")

cluster = importlib.util.module_from_spec(cluster_spec)
cluster_spec.loader.exec_module(cluster)

fraud_w_classes = cluster.fraud_w_classes
fraud_w_classes = fraud_w_classes.sample(frac=1)[:50000]

rand_dim = 32  # 32 # needs to be ~data_dim
base_n_count = 128  # 128

nb_steps = 6000 + 1  # 50000 # Add one for logging of the last interval
batch_size = 256  # 64

k_d = 5  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 100  # 100  # number of steps to pre-train the critic before starting adversarial training
log_interval = 1000  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
learning_rate = 1e-3  # 5e-5
data_dir = 'C:/Users/amarek/PycharmProjects/data_lab/data generation/GANs/models/WcGAN/ori/'
generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None

# show = False
show = False

# load the original data
file_name = 'C:/Users/amarek/PycharmProjects/data_lab/datasets/creditcard_normalised.csv'
ori_data = pd.read_csv(file_name)

fraud_data = ori_data.loc[ori_data['class'] == 1]
X = fraud_data.drop('class', axis=1)
col_names = list(X.columns)

# train the vanilla GAN
arguments = [rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate,
             base_n_count, data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show]

label_cols = [i for i in fraud_w_classes.columns if 'class' in i]
data_cols = [i for i in fraud_w_classes.columns if i not in label_cols]

GAN.adversarial_training_WGAN(arguments, fraud_w_classes, data_cols=data_cols, label_cols=label_cols)  # CGAN

# find the best step
data_dir = 'C:/Users/amarek/PycharmProjects/data_lab/data generation/GANs/models/WCGAN/ori/'
prefix = 'WCGAN'
step = 6000

[combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(
    open(data_dir+prefix+'_losses_step_'+str(step)+'.pkl', 'rb'))

best_step = list(xgb_losses).index(xgb_losses.min()) * 10
print(best_step, xgb_losses.min())

xgb100 = [xgb_losses[i] for i in range(0, len(xgb_losses), 10)]
best_step = xgb100.index(min(xgb100)) * log_interval
print(best_step, min(xgb100))

# Look for the step with the lowest critic loss, and the lowest step saved (every 100)

delta_losses = np.array(disc_loss_real) - np.array(disc_loss_generated)

best_step = list(delta_losses).index(delta_losses.min())
print(best_step, delta_losses.min())

delta100 = [delta_losses[i] for i in range(0, len(delta_losses), 100)]
best_step = delta100.index(min(delta100)) * log_interval
print(best_step, min(delta100))

# # define network models
# data_dim = len(data_cols)
# label_dim = len(label_cols)
# generator_model, discriminator_model, combined_model = GAN.define_models_CGAN(rand_dim, data_dim, label_dim,
#                                                                               base_n_count)
# generator_model.load_weights('C:/Users/amarek/PycharmProjects/data_lab'
#                              '/data generation/GANs/models/WcGAN/WCGAN_generator_model_weights_step_5400.h5')
#
# with_class = True
# if label_dim > 0:
#     with_class = True
# # Now generate some new data
#
# test_size = 3813  # Equal to all of the fraud cases
# train = cluster.fraud_w_classes
# x = GAN.get_data_batch(train, test_size, seed=5)
# z = np.random.normal(size=(test_size, rand_dim))
# if with_class:
#     labels = x[:, -label_dim:]
#     g_z = generator_model.predict([z, labels])
# else:
#     g_z = generator_model.predict(z)
#
# # Check using the same functions used during GAN training
#
# # print(GAN.CheckAccuracy(x, g_z, data_cols, seed=0, with_class=with_class, data_dim=data_dim ) )
#
# # GAN.PlotData(x, g_z, col_names, seed=5, with_class=False, data_dim=data_dim)
#
# df = pd.DataFrame([g_z[0]], columns=fraud_w_classes.columns)
# for i in range(1, len(g_z)):
#     df2 = pd.DataFrame([g_z[i]], columns=fraud_w_classes.columns)
#     df = df.append(df2, ignore_index=True)
#
#
# df['class'] = np.ones(test_size, dtype=np.int)
# print(df)
#
# df.to_pickle('C:/Users/amarek/PycharmProjects/data_lab/datasets/WcGAN5400_ori.pkl')