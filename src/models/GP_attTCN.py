import os
import sys
import tensorflow as tf
import numpy as np
# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.models.GP import MultiKernelMGPLayer
from src.models.attTCN import AttTCN


class GPattTCN:
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
        # a few variables to be used later
        self.tw = time_window
        self.s_feat = n_stat_features
        self.samp = n_mc_samples

        # the model
        self.GP = MultiKernelMGPLayer(time_window=time_window,
                                      n_mc_samples=n_mc_samples,
                                      n_features=n_features,
                                      log_noise_mean=log_noise_mean,
                                      log_noise_std=log_noise_std,
                                      method_name=method_name,
                                      add_diag=add_diag,
                                      save_path=save_path)

        self.attTCN = AttTCN(time_window,
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
        self.n_GP_var = len(self.GP.trainable_variables)

    def __call__(self, inputs):
        self.GP_out = self.GP(inputs[:-1])
        stat_matching_shape = \
            tf.concat([  # step III: concatenate all patients i
                # step II: tile patient info for each MC sample
                tf.tile(

                    # step I: for pat i, repeat feat data for each time step
                    tf.concat([tf.reshape(inputs[-1][i], [1, 1, self.s_feat]) for _ in range(self.tw)], axis=1)
                    , [self.samp, 1, 1])
                for i in range(inputs[-1].shape[0])], axis=0)

        self.TCN_input = tf.concat([self.GP_out, stat_matching_shape], -1)

        return self.attTCN(self.TCN_input)

    def get_weights(self):
        self.trainable_variables = self.GP.trainable_variables + \
                                   self.attTCN.trainable_variables
        return self.trainable_variables

    def set_weights(self, weights):
        if not isinstance(weights[0], np.ndarray):
            weights = [weights[i].numpy() for i in range(len(weights))]
        self.GP.set_weights(weights[:self.n_GP_var])
        self.attTCN.set_weights(weights[self.n_GP_var:])
