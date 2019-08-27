# Adam_l1 is the best
import pandas as pd
import numpy as np
import importlib.util
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from GAN_global_functions import adversarial_training_WGAN, define_models_GAN, get_data_batch, CheckAccuracy, PlotData

cluster_spec = importlib.util.spec_from_file_location("k means classes.py",
                                                      "2) synthetic data generation/cGAN/bioresponse/"
                                                      "k means classes.py")

cluster = importlib.util.module_from_spec(cluster_spec)
cluster_spec.loader.exec_module(cluster)

fraud_w_classes = cluster.fraud_w_classes
data = fraud_w_classes

# for different optimizers check line 424 of GAN_global_functions
# rand_dim needs to be the data dimension
# nb_steps - add one for logging of the last interval
# k_d/k_g number of discriminator/generator network updates per adversarial training step
# critic_pre_train_steps - number of steps to pre-train the critic before starting adversarial training
# log_interval -  interval (in steps) at which to log loss summaries and save plots of image samples to disc
def WcGAN_generate_data(data=data, rand_dim=43, base_n_count=256, nb_steps=5000 + 1, batch_size=128, k_d=5, k_g=1,
                      critic_pre_train_steps=200, log_interval=100, learning_rate=1e-4,
                      data_dir='2) synthetic data generation/WcGAN/bioresponse/WcGAN training/bio_1_v2',
                      gen_data_size=3000, gen_data_name='WcGAN_bio_1_v2_3000'):


    generator_model_path, discriminator_model_path, loss_pickle_path = None, None, None

    # show = False
    show = False
    X = data.drop('class', axis=1)
    col_names = list(X.columns)

    # train the vanilla GAN
    arguments = [rand_dim, nb_steps, batch_size, k_d, k_g, critic_pre_train_steps, log_interval, learning_rate,
                 base_n_count, data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show]

    label_cols = [i for i in fraud_w_classes.columns if 'class' in i]
    data_cols = [i for i in fraud_w_classes.columns if i not in label_cols]

    #adversarial_training_WGAN(arguments, fraud_w_classes, data_cols=data_cols, label_cols=label_cols)  # CGAN

    # find the best training step
    prefix = 'WCGAN'
    last_step = nb_steps-1

    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(
        open(data_dir+prefix+'_losses_step_'+str(last_step)+'.pkl', 'rb'))

    best_step = list(xgb_losses).index(xgb_losses.min()) * 10
    print('best step based on xgb loss', best_step, xgb_losses.min())

    xgb100 = [xgb_losses[i] for i in range(0, len(xgb_losses), int(log_interval/10))]
    best_step_x = xgb100.index(min(xgb100)) * log_interval
    print('best step xgb(based on saved data)', best_step_x, min(xgb100))

    # Look for the step with the lowest discriminator loss, and the lowest step saved (every 100)
    delta_losses = np.array(disc_loss_real) - np.array(disc_loss_generated)

    best_step = list(delta_losses).index(delta_losses.min())
    print('best step discrimnator loss', best_step, delta_losses.min())

    delta100 = [delta_losses[i] for i in range(0, len(delta_losses), log_interval)]
    best_step = delta100.index(min(delta100)) * log_interval
    print('best step disc loss(based on saved data)', best_step, min(delta100))

    # define network models
    data_dim = len(col_names)
    label_dim = len(col_names)
    generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count)
    generator_model.load_weights(data_dir + 'WCGAN_generator_model_weights_step_' + str(best_step_x) + '.h5')

    with_class = False
    if label_dim > 0:
        with_class = True

    # Now generate some new data

    # generate new data (3813,492)
    train = data.drop('class', axis=1)
    x = get_data_batch(train, gen_data_size, seed=5)
    z = np.random.normal(size=(gen_data_size, rand_dim))
    g_z = generator_model.predict(z)

    # Check using the same functions used during GAN training

    # print(CheckAccuracy(x, g_z, col_names, seed=0, with_class=with_class, data_dim=data_dim ) )

    # PlotData(x, g_z, col_names, seed=5, with_class=False, data_dim=data_dim)

    df = pd.DataFrame([g_z[0]], columns=col_names)
    for i in range(1, len(g_z)):
        df2 = pd.DataFrame([g_z[i]], columns=col_names)
        df = df.append(df2, ignore_index=True)

    df['class'] = np.ones(gen_data_size, dtype=np.int)

    df.to_pickle('2) synthetic data generation/WcGAN/bioresponse/WcGAN results/'+gen_data_name+'.pkl')

    plt.plot(np.transpose([range(0,nb_steps,1)]),disc_loss_generated, label='discriminator loss on fake')
    plt.plot(np.transpose([range(0, nb_steps, 1)]), disc_loss_real, label='discriminator loss on real')
    plt.plot(np.transpose([range(0,nb_steps,1)]),combined_loss, label='generator loss')
    plt.legend()
    plt.title('WcGAN training - bioresponse dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.xticks(np.arange(0,nb_steps, step=log_interval))
    plt.savefig('2) synthetic data generation/WcGAN/bioresponse/WcGAN results/'+gen_data_name+'.png',
                bbox_inches='tight')
    # plt.show()

    with open('2) synthetic data generation/WcGAN/bioresponse/WcGAN results/'+gen_data_name+'.txt','w')as a:
        a.write(data_dir+'\n'+'best xboost step(used for data generation):'+str(best_step_x)+' '+str(min(xgb100))+'\n'+
                'best step for delta losses:'+str(best_step)+' '+str(min(delta100))+'\n'+'base_n_count:'+str(base_n_count)
                +'\n'+'nb_steps:'+ str(nb_steps)+'\n'+'batch_size:'+str(batch_size)+'\n'+'critic_pre_train_steps:'+
                str(critic_pre_train_steps)+'\n'+'log_interval:'+str(log_interval)+'\n'+'learning_rate:'+
                str(learning_rate)+'\n'+'gen_data_size:'+str(gen_data_size))

    return

WcGAN_generate_data()