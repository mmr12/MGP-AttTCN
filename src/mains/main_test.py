import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import pickle
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)
from src.utils.debug import t_print
from src.data_loader.loader import DataGenerator
from src.trainers.tester import Tester
# models
from src.models.GP_attTCN import GPattTCN
from src.models.GP_TCN import GP_TCN
from src.models.GP_logreg import GPLogReg
from src.models.GP_attTCN_ablations import GPattTCN_alpha, GPattTCN_beta



def main(hparam_file, model_name, checkpoint_file, log_path):
    # load hpyerparams
    with open(hparam_file, 'rb') as f:
        hparams = pickle.load(f)

    # load model
    if model_name == 'mgpatttcn':
        model = GPattTCN(hparams['time_window'],
                     hparams['no_mc_samples'],
                     hparams['n_features'],
                     hparams['n_stat_features'],
                     kernel=hparams['kernel_choice'],
                     L2reg=hparams['L2reg'],
                     DO=hparams['DO'],
                     num_layers=hparams['num_layers'],
                     kernel_size=hparams['kernel_size'],
                     stride=hparams['stride'],
                     sigmoid_beta=hparams['sigmoid_beta'])
    elif model_name == 'mgptcn':
        model = GP_TCN(hparams['time_window'],
                     hparams['no_mc_samples'],
                     hparams['n_features'],
                     hparams['n_stat_features'],
                     kernel=hparams['kernel_choice'],
                     L2reg=hparams['L2reg'],
                     DO=hparams['DO'],
                     num_layers=hparams['num_layers'],
                     kernel_size=hparams['kernel_size'],
                     stride=hparams['stride'],
                     sigmoid_beta=hparams['sigmoid_beta'])
    elif model_name == 'abl_0':
        model = GPLogReg(hparams['time_window'],
                     hparams['no_mc_samples'],
                     hparams['n_features'],
                     hparams['n_stat_features'],
                     L2reg=hparams['L2reg'],)
    elif model_name == 'abl_alpha':
        model = GPattTCN_alpha(hparams['time_window'],
                     hparams['no_mc_samples'],
                     hparams['n_features'],
                     hparams['n_stat_features'],
                     kernel=hparams['kernel_choice'],
                     L2reg=hparams['L2reg'],
                     DO=hparams['DO'],
                     num_layers=hparams['num_layers'],
                     kernel_size=hparams['kernel_size'],
                     stride=hparams['stride'],
                     sigmoid_beta=hparams['sigmoid_beta'])
    elif model_name == 'abl_beta':
        model = GPattTCN_beta(hparams['time_window'],
                     hparams['no_mc_samples'],
                     hparams['n_features'],
                     hparams['n_stat_features'],
                     kernel=hparams['kernel_choice'],
                     L2reg=hparams['L2reg'],
                     DO=hparams['DO'],
                     num_layers=hparams['num_layers'],
                     kernel_size=hparams['kernel_size'],
                     stride=hparams['stride'],
                     sigmoid_beta=hparams['sigmoid_beta'])
    else: raise NameError

    # load checkpoint
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    model.set_weights(checkpoint['weights'])

    # load data
    data = DataGenerator(no_mc_samples=hparams['no_mc_samples'],
                         max_no_dtpts=hparams['max_no_dtpts'],
                         min_no_dtpts=hparams['min_no_dtpts'],
                         batch_size=hparams['batch_size'],
                         fast_load=False,
                         to_save=True,
                         debug=True,
                         fixed_idx_per_class=False,
                         features=hparams['features'])

    tester = Tester(model,
                 data,
                 hparams['batch_size'],
                 log_path,)

    tester.run()

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--epoch_numpber', type=int)
    parser.add_argument('--model_name', type=int)
    args = parser.parse_args()
    path = os.path.join(head, 'logs', args.exp_name)
    h_param_file = os.path.join(path, 'hyperparam.pkl')
    checkpoint_file = os.path.join(path, 'epoch_{}_out.pkl'.format(args.epoch_number))
    log_path = os.path.join(path, 'epoch_{}_test.pkl'.format(args.epoch_number))
    main(h_param_file, args.model_name, checkpoint_file, log_path)