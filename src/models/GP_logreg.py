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


class GPLogReg:
    def __init__(self,
                 time_window,
                 n_mc_samples,
                 n_features,
                 n_stat_features,
                 log_noise_mean=-2,
                 log_noise_std=0.1,
                 method_name='chol',
                 add_diag=0.001,
                 L2reg=None,
                 save_path=head,
                 ):
        # a few variables to be used later
        self.tw = time_window
        self.non_s_feat = n_features
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

        self.LogReg = tf.keras.Sequential(
            [tf.keras.layers.Dense(2,
                                   input_shape=(time_window * n_features + n_stat_features,),
                                   kernel_regularizer=tf.keras.regularizers.L2(L2reg[0]),
                                   bias_regularizer=tf.keras.regularizers.L2(L2reg[0]),)])

        self.trainable_variables = self.GP.trainable_variables + \
                                   self.LogReg.trainable_variables
        self.n_GP_var = len(self.GP.trainable_variables)

    def __call__(self, inputs):
        self.GP_out = self.GP(inputs[:-1])
        # GP out: batch x MC samples x tw x features
        self.GP_out = tf.reshape(self.GP_out, (-1, self.samp, self.tw * self.non_s_feat))
        stat_input = tf.expand_dims(inputs[-1], axis=1)
        stat_input = tf.broadcast_to(stat_input, [stat_input.shape[0], self.samp, stat_input.shape[-1]])
        self.LR_input = tf.concat([self.GP_out, stat_input], axis=-1)
        self.LR_input = tf.reshape(self.LR_input, (-1, self.tw * self.non_s_feat + self.s_feat))

        return self.LogReg(self.LR_input)

    def get_weights(self):
        self.trainable_variables = self.GP.trainable_variables + \
                                   self.LogReg.trainable_variables
        return self.trainable_variables

    def set_weights(self, weights):
        if not isinstance(weights[0], np.ndarray):
            weights = [weights[i].numpy() for i in range(len(weights))]
        self.GP.set_weights(weights[:self.n_GP_var])
        self.LogReg.set_weights(weights[self.n_GP_var:])
