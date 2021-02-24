import os
import sys
import tensorflow as tf
import numpy as np
# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.models.GP_attTCN import GPattTCN
from src.models.attTCN_alpha import AttTCN_alpha
from src.models.attTCN_beta import AttTCN_beta

class GPattTCN_alpha(GPattTCN):
    def __init__(self,
                 time_window,
                 n_mc_samples,
                 n_features,
                 n_stat_features,
                 kernel='OU',
                 len_mode='avg',
                 len_trainable=True,
                 log_noise_mean=-2,
                 log_noise_std=0.1,
                 log_length_mean=1,
                 log_length_std=0.1,
                 method_name='chol',
                 add_diag=0.001,
                 L2reg=None,
                 DO=None,
                 save_path=head,
                 num_layers=4,
                 kernel_size=2,
                 stride=1,
                 sigmoid_beta=False,
                 moor_data=False
                 ):
        super().__init__(time_window,
                         n_mc_samples,
                         n_features,
                         n_stat_features,
                         kernel,
                         len_mode,
                         len_trainable,
                         log_noise_mean,
                         log_noise_std,
                         log_length_mean,
                         log_length_std,
                         method_name,
                         add_diag,
                         L2reg,
                         DO,
                         save_path,
                         num_layers,
                         kernel_size,
                         stride,
                         sigmoid_beta,
                         moor_data)
        self.attTCN = AttTCN_alpha(time_window,
                             n_features + n_stat_features,
                             num_layers,
                             DO,
                             L2reg,
                             kernel_size=kernel_size,
                             stride=stride,
                             )

        self.trainable_variables = self.GP.trainable_variables + \
                                   self.attTCN.trainable_variables


class GPattTCN_beta(GPattTCN):
    def __init__(self,
                 time_window,
                 n_mc_samples,
                 n_features,
                 n_stat_features,
                 kernel='OU',
                 len_mode='avg',
                 len_trainable=True,
                 log_noise_mean=-2,
                 log_noise_std=0.1,
                 log_length_mean=1,
                 log_length_std=0.1,
                 method_name='chol',
                 add_diag=0.001,
                 L2reg=None,
                 DO=None,
                 save_path=head,
                 num_layers=4,
                 kernel_size=2,
                 stride=1,
                 sigmoid_beta=False,
                 moor_data=False
                 ):
        super().__init__(time_window,
                         n_mc_samples,
                         n_features,
                         n_stat_features,
                         kernel,
                         len_mode,
                         len_trainable,
                         log_noise_mean,
                         log_noise_std,
                         log_length_mean,
                         log_length_std,
                         method_name,
                         add_diag,
                         L2reg,
                         DO,
                         save_path,
                         num_layers,
                         kernel_size,
                         stride,
                         sigmoid_beta,
                         moor_data)
        self.attTCN = AttTCN_beta(time_window,
                             n_features + n_stat_features,
                             num_layers,
                             DO,
                             L2reg,
                             kernel_size=kernel_size,
                             stride=stride,
                             sigmoid_beta=sigmoid_beta
                             )

        self.trainable_variables = self.GP.trainable_variables + \
                                   self.attTCN.trainable_variables