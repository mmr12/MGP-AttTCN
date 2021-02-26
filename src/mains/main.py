import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import pickle
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import t_print 
from src.data_loader.loader import DataGenerator
from src.models.GP_attTCN import GPattTCN
from src.trainers.trainer import Trainer
from sacred import Experiment

exp_name = 'GP_AttTCN'
ex = Experiment(exp_name)
#tf.enable_eager_execution()


@ex.config
def my_config():
    # data
    max_no_dtpts = 250  # chopping 4.6% of data at 250
    min_no_dtpts = 40  # helping with covariance singularity
    time_window = 25  # fixed
    n_features = 24  # old data: 44
    n_stat_features = 8  # old data: 35
    features = 'mr_features_mm_labels'
    n_features=17
    n_stat_features=8
    features = None
    late_patients_only = False
    horizon0 = False

    # model
    model_choice = 'Att'  # ['Att', 'Moor']

    # MGP
    no_mc_samples = 10
    kernel_choice = 'OU'

    # TCN
    num_layers = 4
    kernel_size = 3
    stride = 1
    DO = [0.01] * 10
    L2reg = [0.000001] * 10
    sigmoid_beta = True

    # training
    learning_rate = 0.0005
    batch_size = 64
    num_epochs = 100

@ex.config
def random_search_config():
    num_layers = np.random.randint(2, high=10, size=None, dtype='l')
    learning_rate = np.random.uniform(10e-6, high=10e-4, size=None)
    no_mc_samples = np.random.randint(4, high=20, size=None, dtype='l')
    #DO = [np.random.uniform(0, high=0.99, size=None) for _ in range(num_layers)]
    #L2reg = [np.random.uniform(0, high=250, size=None) for _ in range(num_layers)]
    load_path = head + "/not_a_path"
    kernel_size = np.random.randint(2, high=6, size=None, dtype='l')



@ex.automain
def main(
        # data
        max_no_dtpts,
        min_no_dtpts,
        time_window,
        n_features,
        n_stat_features,
        features,
        late_patients_only,
        horizon0,
        # model
        model_choice,
        # MGP
        no_mc_samples,
        kernel_choice,
        # TCN
        num_layers,
        kernel_size,
        stride,
        DO,
        L2reg,
        sigmoid_beta,
        # training
        learning_rate,
        batch_size,
        num_epochs,
        # sacred
        _run):
    # generate save path
    logdir = os.path.join( "logs/", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    Dict = {
        # data
        "max_no_dtpts":max_no_dtpts,
        "min_no_dtpts":min_no_dtpts,
        "time_window": time_window,
        "n_features": n_features,
        "n_stat_features": n_stat_features,
        "features": features,
        "late_patients_only": late_patients_only,
        "horizon0": horizon0,
        # model
       "model_choice": model_choice,
        # MGP
        "no_mc_samples": no_mc_samples,
        "kernel_choice": kernel_choice,
        # TCN
        "num_layers": num_layers,
        "kernel_size": kernel_size,
        "stride": stride,
        "DO": DO,
        "L2reg": L2reg,
        "sigmoid_beta": sigmoid_beta,
        # training
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }
    with open(os.path.join(logdir, 'hyperparam.pkl'), "wb") as f:
        pickle.dump(Dict, f)

    summary_writers = {'train': tf.summary.create_file_writer(os.path.join(logdir, 'train')),
                       'val': tf.summary.create_file_writer(os.path.join(logdir, 'val')),
                       }
    t_print("nu_layers: {}\tlr: {}\tMC samples :{}\tDO :{}\tL2 :{}\t kernel:{}".format(num_layers, learning_rate, no_mc_samples, DO[0], L2reg[0], kernel_size))
    # Load data
    data = DataGenerator(no_mc_samples=no_mc_samples,
                         max_no_dtpts=max_no_dtpts,
                         min_no_dtpts=min_no_dtpts,
                         batch_size=batch_size,
                         fast_load=False,
                         to_save=True,
                         debug=True,
                         fixed_idx_per_class=False,
                         features=features)

    t_print("main - generate model and optimiser")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_step = tf.Variable(0)

    # Load model
    model = GPattTCN(time_window,
                     no_mc_samples,
                     n_features,
                     n_stat_features,
                     kernel=kernel_choice,
                     L2reg=L2reg,
                     DO=DO,
                     num_layers=num_layers,
                     kernel_size=kernel_size,
                     stride=stride,
                     sigmoid_beta=sigmoid_beta)

    # Initialise trainer
    trainer = Trainer(model=model,
                      data=data,
                      num_epochs=num_epochs,
                      batch_size=batch_size,
                      optimizer=optimizer,
                      global_step=global_step,
                      _run=_run,
                      summary_writers=summary_writers,
                      log_path=logdir,
                      train_only=False,
                      notebook_friendly=False,
                      eval_every=20,
                      late_patients_only=late_patients_only,
                      horizon0=horizon0,)

    # train model
    trainer.run()
    ignore = True
    if not ignore:
        n_pts = len(data.val_data[-1])
        y = np.reshape(np.asarray(trainer.all_dev_y), (-1))
        y_hat = np.reshape(np.asarray(trainer.all_dev_y_hat), (-1, 2))[:, 0]
        for i in range(n_pts):
            _run.log_scalar("y_dev", y[i])
            _run.log_scalar("y_hat_dev", y_hat[i])
