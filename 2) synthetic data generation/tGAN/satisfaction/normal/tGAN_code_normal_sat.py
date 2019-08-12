# https://github.com/DAI-Lab/TGAN
# Lei Xu, Kalyan Veeramachaneni. 2018. Synthesizing Tabular Data using Generative Adversarial Networks

import numpy as np
import pandas as pd
from tgan.model import TGANModel

file_name = 'data/satisfaction/satisfaction clean.pkl'
ori_data = pd.read_pickle(file_name)
fraud_data = ori_data.loc[ori_data['class'] == 0]

# use only columns which are not in the list of categorical columns -> are continous
continuous_columns = [x for x in fraud_data.columns if x not in ['ind_var1_0', 'ind_var1', 'ind_var5_0', 'ind_var5', 'ind_var6_0', 'ind_var6', 'ind_var8_0', 'ind_var8',
        'ind_var12_0', 'ind_var12', 'ind_var13_0', 'ind_var13_corto_0', 'ind_var13_corto', 'ind_var13_largo_0',
        'ind_var13_largo', 'ind_var13_medio_0', 'ind_var13', 'ind_var14_0', 'ind_var14', 'ind_var17_0', 'ind_var17',
        'ind_var18_0', 'ind_var19', 'ind_var20_0', 'ind_var20', 'ind_var24_0', 'ind_var24', 'ind_var25_cte',
        'ind_var26_0', 'ind_var26_cte', 'ind_var25_0', 'ind_var30_0', 'ind_var30', 'ind_var31_0', 'ind_var31',
        'ind_var32_cte', 'ind_var32_0', 'ind_var33_0', 'ind_var33', 'ind_var34_0', 'ind_var37_cte', 'ind_var37_0',
        'ind_var39_0', 'ind_var40_0', 'ind_var40', 'ind_var41_0', 'ind_var44_0', 'ind_var44', 'ind_var7_emit_ult1',
        'ind_var7_recib_ult1', 'ind_var10_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var9_ult1',
        'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'class']]

tgan = TGANModel(continuous_columns, output='2) synthetic data generation/tGAN/satisfaction/normal/', max_epoch=1,
                 steps_per_epoch=6000, save_checkpoints=True,
                 restore_session=True, batch_size=256, z_dim=200, noise=0.2, l2norm=0.00001, learning_rate=0.001,
                 num_gen_rnn=100, num_gen_feature=100, num_dis_layers=1, num_dis_hidden=100, optimizer='AdamOptimizer')

tgan.fit(fraud_data)
model_path = '2) synthetic data generation/tGAN/satisfaction/normal/tGAN_normal_model.pkl'
tgan.save(model_path, force=True) #force=True to overwrite

model_path = '2) synthetic data generation/tGAN/satisfaction/normal/tGAN_normal_model.pkl'
loaded_tgan = TGANModel.load(model_path)

num_samples = len(fraud_data)
samples = loaded_tgan.sample(num_samples)
samples.to_pickle('2) synthetic data generation/tGAN/satisfaction/normal/tGAN_normal_sat_73012.pkl')


# #!usr/bin/env python
#
# """Tune and evaluate TGAN models."""
# import json
# import os
#
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorpack.utils import logger
#
# from tgan.model import TUNABLE_VARIABLES, TGANModel
# from tgan.research.evaluation import evaluate_classification
#
#
# def prepare_hyperparameter_search(epoch, steps_per_epoch, num_random_search):
#     """Prepare hyperparameters."""
#     model_kwargs = []
#     basic_kwargs = {
#         'max_epoch': epoch,
#         'steps_per_epoch': steps_per_epoch,
#     }
#
#     for i in range(num_random_search):
#         kwargs = {name: np.random.choice(choices) for name, choices in TUNABLE_VARIABLES.items()}
#         kwargs.update(basic_kwargs)
#         model_kwargs.append(kwargs)
#
#     return model_kwargs
#
#
# def fit_score_model(
#         name, model_kwargs, train_data, test_data, continuous_columns,
#         sample_rows, store_samples
# ):
#     """Fit and score models using given params."""
#     for index, kwargs in enumerate(model_kwargs):
#         logger.info('Training TGAN Model %d/%d', index + 1, len(model_kwargs))
#
#         tf.reset_default_graph()
#         base_dir = os.path.join('experiments', name)
#         output = os.path.join(base_dir, 'model_{}'.format(index))
#         model = TGANModel(continuous_columns, output=output, **kwargs)
#         model.fit(train_data)
#         sampled_data = model.sample(sample_rows)
#
#         if store_samples:
#             dir_name = os.path.join(base_dir, 'data')
#             if not os.path.isdir(dir_name):
#                 os.mkdir(dir_name)
#
#             file_name = os.path.join(dir_name, 'model_{}.csv'.format(index))
#             sampled_data.to_csv(file_name, index=False, header=True)
#
#         score = evaluate_classification(sampled_data, test_data, continuous_columns)
#         model_kwargs[index]['score'] = score
#
#     return model_kwargs
#
#
# def run_experiment(
#     name, epoch, steps_per_epoch, sample_rows, train_csv, continuous_cols,
#     num_random_search, store_samples=True, force=False
# ):
#     """Run experiment using the given params and collect the results.
#     The experiment run the following steps:
#     1. We fetch and split our data between test and train.
#     2. We first train a TGAN data synthesizer using the real training data T and generate a
#        synthetic training dataset Tsynth.
#     3. We then train machine learning models on both the real and synthetic datasets.
#     4. We use these trained models on real test data and see how well they perform.
#     """
#     if os.path.isdir(name):
#         if force:
#             logger.info('Folder "{}" exists, and force=True. Deleting folder.'.format(name))
#             os.rmdir(name)
#
#         else:
#             raise ValueError(
#                 'Folder "{}" already exist. Please, use force=True to force deletion '
#                 'or use a different name.'.format(name))
#
#     # Load and split data
#     data = pd.read_csv(train_csv, header=-1)
#     train_data, test_data = train_test_split(data, train_size=0.8)
#
#     # Prepare hyperparameter search
#     model_kwargs = prepare_hyperparameter_search(epoch, steps_per_epoch, num_random_search)
#
#     return fit_score_model(
#         name, model_kwargs, train_data, test_data,
#         continuous_cols, sample_rows, store_samples
#     )
#
#
# def numpy_default(obj):
#     """Change numpy objects into json-serializable ones."""
#     if isinstance(obj, (np.int64, np.int32, np.int16)):
#         return int(obj)
#
#     if isinstance(obj, [np.float64]):
#         return float(obj)
#
#     raise TypeError
#
#
# def run_experiments(config_path, output_path):
#     """Run experiments specified in JSON file."""
#     with open(config_path) as f:
#         experiments_config = json.load(f)
#
#     result = {}
#     if isinstance(experiments_config, list):
#         for experiment in experiments_config:
#             name = experiment['name']
#             result[name] = run_experiment(**experiment)
#     else:
#         name = experiments_config['name']
#         result[name] = run_experiment(**experiments_config)
#
#     with open(output_path, 'w') as f:
#         json.dump(result, f, default=numpy_default)
#
# # run_experiments('C:/Users/amarek/PycharmProjects/data_lab/data generation/GANs/models/TGAN/json/setup',
##                 'C:/Users/amarek/PycharmProjects/data_lab/data generation/GANs/models/TGAN/json/')