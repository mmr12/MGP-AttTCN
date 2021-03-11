import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import pickle
from argparse import ArgumentParser
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import t_print
from src.data_loader.loader import DataGenerator
from src.models.GP_attTCN_ablations import GPattTCN_alpha
from src.trainers.trainer import Trainer


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
        seed):
    # generate save path
    logdir = os.path.join("logs/abl_alpha", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    Dict = {
        # data
        "max_no_dtpts": max_no_dtpts,
        "min_no_dtpts": min_no_dtpts,
        "time_window": time_window,
        "n_features": n_features,
        "n_stat_features": n_stat_features,
        "features": features,
        "late_patients_only": late_patients_only,
        "horizon0": horizon0,
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
        "seed": seed
    }
    with open(os.path.join(logdir, 'hyperparam.pkl'), "wb") as f:
        pickle.dump(Dict, f)

    summary_writers = {'train': tf.summary.create_file_writer(os.path.join(logdir, 'train')),
                       'val': tf.summary.create_file_writer(os.path.join(logdir, 'val')),
                       'val_hz': tf.summary.create_file_writer(os.path.join(logdir, 'val_hz'))
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
    model = GPattTCN_alpha(time_window,
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
                      summary_writers=summary_writers,
                      log_path=logdir,
                      train_only=False,
                      notebook_friendly=False,
                      eval_every=20,
                      late_patients_only=late_patients_only,
                      horizon0=horizon0,)

    # train model
    trainer.run()

if __name__=="__main__":
    tf.random.set_seed(1237)
    np.random.seed(1237)
    # data
    max_no_dtpts = 250  # chopping 4.6% of data at 250
    min_no_dtpts = 40  # helping with covariance singularity
    time_window = 25  # fixed
    n_features= 17
    n_stat_features= 8
    late_patients_only = False
    horizon0 = False

    # model
    model_choice = 'Att'  # ['Att', 'Moor']

    # MGP
    kernel_choice = 'OU'

    # TCN
    stride = 1
    DO = [0.01] * 10
    sigmoid_beta = True

    # training
    batch_size = 128
    num_epochs = 100

    parser = ArgumentParser()
    parser.add_argument('--learning_rate',
                        default=np.random.uniform(10e-6, high=10e-4, size=None),
                        type=float)
    parser.add_argument('--no_mc_samples',
                        default=np.random.randint(8, high=20, size=None, dtype='l'),
                        type=int)
    parser.add_argument('--L2reg', default=np.random.randint(-5, high=8, size=None, dtype='l'), type=float)
    parser.add_argument('--kernel_size', default=np.random.randint(2, high=6, size=None, dtype='l'), type=int)
    parser.add_argument('--num_layers', default=np.random.randint(2, high=8, size=None, dtype='l'), type=int)
    parser.add_argument('--seed', default=np.random.randint(1, high=9999, size=None, dtype='l'), type=int)
    parser.add_argument('--features', default='rosnati', type=str)
    args = parser.parse_args()
    learning_rate = args.learning_rate
    no_mc_samples = args.no_mc_samples
    kernel_size = (args.kernel_size,)
    num_layers = args.num_layers
    seed = args.seed
    features = args.features
    tf.random.set_seed(seed)
    np.random.seed(seed)
    L2reg = [10**float(args.L2reg)] * 10
    main(
        # data
        max_no_dtpts,
        min_no_dtpts,
        time_window,
        n_features,
        n_stat_features,
        features,
        late_patients_only,
        horizon0,
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
        seed)
