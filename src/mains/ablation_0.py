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
from src.models.GP_logreg import GPLogReg
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
        # model
        model_choice,
        # MGP
        no_mc_samples,
        kernel_choice,
        L2reg,
        # training
        learning_rate,
        batch_size,
        num_epochs,):
    # generate save path
    logdir = os.path.join("logs/ablation_0", datetime.now().strftime("%Y%m%d-%H%M%S"))
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
        "L2reg": L2reg,
        # training
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }
    with open(os.path.join(logdir, 'hyperparam.pkl'), "wb") as f:
        pickle.dump(Dict, f)

    summary_writers = {'train': tf.summary.create_file_writer(os.path.join(logdir, 'train')),
                       'val': tf.summary.create_file_writer(os.path.join(logdir, 'val'))}
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
    model = GPLogReg(time_window,
                 no_mc_samples,
                 n_features,
                 n_stat_features,
                 L2reg=L2reg)

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
    n_features = 24  # old data: 44
    n_stat_features = 8  # old data: 35
    features = 'mr_features_mm_labels'
    n_features= 17
    n_stat_features= 8
    features = None
    late_patients_only = False
    horizon0 = False

    # model
    model_choice = 'Att'  # ['Att', 'Moor']

    # MGP
    no_mc_samples = 10
    kernel_choice = 'OU'

    # training
    learning_rate = 0.0005
    batch_size = 128
    num_epochs = 100

    learning_rate = np.random.uniform(10e-6, high=10e-4, size=None)
    no_mc_samples = np.random.randint(8, high=20, size=None, dtype='l')
    L2reg = [10**float(np.random.randint(-5, high=8, size=None, dtype='l'))] * 5


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
        # model
        model_choice,
        # MGP
        no_mc_samples,
        kernel_choice,
        L2reg,
        # training
        learning_rate,
        batch_size,
        num_epochs,)
