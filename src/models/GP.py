#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)
from src.models.GP_utils import kroneker_matrix, OU_kernel, K_vitals_initialiser, K_labs_initialiser
from src.utils.debug import t_print


class MultiKernelMGPLayer(tf.keras.layers.Layer):
    def __init__(self,
                 time_window,
                 n_mc_samples,
                 n_features,
                 log_noise_mean=-2,
                 log_noise_std=0.1,
                 log_length_v_mean=np.log(0.5),
                 log_length_l_mean=np.log(10),
                 log_length_std=0.001,
                 method_name='chol',
                 add_diag=0.001,
                 save_path=head,
                 moor_data=False):

        super(MultiKernelMGPLayer, self).__init__()
        if method_name not in ['chol', 'cg']:
            raise NameError("method_name not in ['chol', 'cg']")
        # t_print("Welcome to MyMGPLayer")
        # number of Monte Carlo samples
        self.time_window = time_window
        self.n_mc_samples = n_mc_samples
        self.save_path = save_path

        # number of medical features
        self.n_features = n_features

        # covariance of medical features
        # enforcing positive definitiveness by using its cholesky decomposition
        self.K_D_half_v_prior = self.add_weight(name="GP_features_kernel",
                                                  shape=[n_features, n_features],
                                                  initializer=K_vitals_initialiser,
                                                  trainable=True
                                                  )

        self.K_D_half_l_prior = self.add_weight(name="GP_features_kernel",
                                                  shape=[n_features, n_features],
                                                  initializer=K_labs_initialiser,
                                                  trainable=True
                                                  )

        # noise associated to each feature reading
        self.log_noises = self.add_weight(name="GP_log_noises",
                                            shape=[n_features],
                                            initializer=tf.initializers.truncated_normal(mean=log_noise_mean,
                                                                                         stddev=log_noise_std),
                                            trainable=True
                                            )

        # time covariance kernel parameter
        self.log_length_v = self.add_weight(name="GP_log_length",
                                              shape=[1],
                                              initializer=tf.initializers.truncated_normal(mean=log_length_v_mean,
                                                                                           stddev=log_length_std),
                                              trainable=True
                                              )

        self.log_length_l = self.add_weight(name="GP_log_length",
                                              shape=[1],
                                              initializer=tf.initializers.truncated_normal(mean=log_length_l_mean,
                                                                                           stddev=log_length_std),
                                              trainable=True
                                              )

        # matrix decomposition method
        self.method_name = method_name
        self.add_diag = add_diag
        self.lost_to_OOM = []
        self.moor_data = moor_data

    def variable_update(self):
        # K_D from K_D_half_prior
        # keep lower triangular part
        self.K_D_v_half = tf.linalg.band_part(self.K_D_half_v_prior, -1, 0)
        self.K_D_l_half = tf.linalg.band_part(self.K_D_half_l_prior, -1, 0)
        # multiply
        self.K_D_v = tf.matmul(self.K_D_v_half, tf.transpose(self.K_D_v_half))
        self.K_D_l = tf.matmul(self.K_D_l_half, tf.transpose(self.K_D_l_half))

        # D from log_noises
        noises = tf.exp(self.log_noises)
        self.D = tf.linalg.diag(noises)

        # length from log_length
        self.length_v = tf.exp(self.log_length_v)
        self.length_l = tf.exp(self.log_length_l)

    def __call__(self, inputs):
        # first calculate data indep. (but differentiable) matrices
        # this needs to happen within 'call' to be taped in the GradientTape (TF eager exec)
        self.variable_update()

        if len(inputs[0].shape) == 1:
            for k in range(len(inputs)):
                inputs[k] = tf.reshape(inputs[k], (-1, 1))
        # Y, T, ind_features, num_distinct_Y, X, num_distinct_X,
        Y, T, ind_K_D, ind_T, num_obs, X, num_tcn_grid_times = inputs

        grid_max = self.time_window
        Z = tf.zeros((0, grid_max, self.n_features))
        batch_size = tf.shape(T)[0]

        # while loop is parallelisable
        def cond(i, Z):
            return i < batch_size

        def body(i, Z):
            Yi = tf.reshape(tf.slice(Y, [i, 0], [1, num_obs[i]]), [-1])
            Ti = tf.reshape(tf.slice(T, [i, 0], [1, num_obs[i]]), [-1])
            ind_K_Di = tf.reshape(tf.slice(ind_K_D, [i, 0], [1, num_obs[i]]), [-1])
            ind_Ti = tf.reshape(tf.slice(ind_T, [i, 0], [1, num_obs[i]]), [-1])
            Xi = tf.reshape(tf.slice(X, [i, 0], [1, num_tcn_grid_times[i]]), [-1])
            X_len = num_tcn_grid_times[i]
            try:
                GP_draws_i = self.draw_GP(Yi=Yi,
                                          Ti=Ti,
                                          ind_K_Di=ind_K_Di,
                                          ind_Ti=ind_Ti,
                                          Xi=Xi,
                                          X_len=X_len,
                                          i=i)
                padding = grid_max - X_len
                if padding > 0:
                    # aligning data to the right
                    padded_GP_draws_i = tf.concat([tf.zeros((self.n_mc_samples, padding, self.n_features)), GP_draws_i],
                                                  1)
                elif padding < 0:
                    padded_GP_draws_i = tf.slice(GP_draws_i,
                                                 [0, GP_draws_i.shape[1] - grid_max, 0],
                                                 [self.n_mc_samples, grid_max, self.n_features])
                else:
                    padded_GP_draws_i = GP_draws_i
                Z = tf.concat([Z, padded_GP_draws_i], 0)
                # t_print("Z: {}".format(Z.shape))
            except Exception as e:
                print(e)
                self.lost_to_OOM.append([Yi, Ti, ind_K_Di, ind_Ti, Xi, X_len, self.length, self.K_D, self.D])
                Z = tf.concat([Z, tf.zeros((self.n_mc_samples, grid_max, self.n_features))], 0)
            return i + 1, Z

        i = tf.constant(0)
        (i, Z) = tf.while_loop(cond,
                               body,
                               loop_vars=[i, Z],
                               shape_invariants=[i.get_shape(),
                                                 tf.TensorShape([None, None, None])])

        return Z

    def draw_GP(self,
                Yi,
                Ti,
                ind_K_Di,
                ind_Ti,
                Xi,
                X_len,
                i
                ):

        # step I: calculate Sigma = K_D_v x_kroneker K_Ti_v + K_D_l x_kroneker K_Ti_l + D x_kroneker I
        Ti_big = tf.gather(Ti, ind_Ti)

        K_Ti_v_big = OU_kernel(length=self.length_v, x1=Ti_big, x2=Ti_big)
        K_Ti_l_big = OU_kernel(length=self.length_l, x1=Ti_big, x2=Ti_big)

        # calculate the kroneker product only for values that are actually present
        K_D_v_big = kroneker_matrix(self.K_D_v, ind_K_Di)
        K_D_l_big = kroneker_matrix(self.K_D_l, ind_K_Di)

        K_D__K_Ti = tf.multiply(K_D_v_big, K_Ti_v_big) + tf.multiply(K_D_l_big, K_Ti_l_big)

        D_big = kroneker_matrix(self.D, ind_K_Di)
        D__I = tf.linalg.diag(tf.linalg.diag_part(D_big))

        Sigma_prior = K_D__K_Ti + D__I + self.add_diag * tf.eye(tf.cast(D__I.shape[0], tf.int32))

        # step II: calculate Sigma for new datapoints
        ind_K_D_l = tf.concat([tf.tile([i], [X_len]) for i in range(self.n_features)], 0)
        K_D_v_big = kroneker_matrix(self.K_D_v, ind_K_D_l)
        K_D_l_big = kroneker_matrix(self.K_D_l, ind_K_D_l)
        ind_K_Xi = tf.tile(tf.range(X_len), [self.n_features])
        Xi_big = tf.gather(Xi, ind_K_Xi)
        K_Xi_v_big = OU_kernel(self.length_v, Xi_big, Xi_big)
        K_Xi_l_big = OU_kernel(self.length_l, Xi_big, Xi_big)
        K_D__K_Xi = tf.multiply(K_D_v_big, K_Xi_v_big) + tf.multiply(K_D_l_big, K_Xi_l_big)

        # K_D__K_XT
        K_D_v_big = kroneker_matrix(self.K_D_v, ind_K_Di, ind_K_D_l)
        K_D_l_big = kroneker_matrix(self.K_D_l, ind_K_Di, ind_K_D_l)
        K_XT_v_big = OU_kernel(self.length_v, Xi_big, Ti_big)
        K_XT_l_big = OU_kernel(self.length_l, Xi_big, Ti_big)
        K_D__K_XT = tf.multiply(K_D_v_big, K_XT_v_big) + tf.multiply(K_D_l_big, K_XT_l_big)
        # t_print("K_D__K_XT {}".format(K_D__K_XT.shape))

        # K_D_K_TX
        K_D__K_TX = tf.transpose(K_D__K_XT)

        # step III: inverse Sigma_prior
        L, num_tries = self.try_cholesky(Sigma_prior)
        if self.moor_data:
            Mu = tf.matmul(K_D__K_XT, tf.linalg.cholesky_solve(L, tf.reshape(Yi, [-1, 1])))
        else:
            Yi_reordered = tf.gather(Yi, ind_Ti)
            Mu = tf.matmul(K_D__K_XT, tf.linalg.cholesky_solve(L, tf.reshape(Yi_reordered, [-1, 1])))
        Sigma = K_D__K_Xi - tf.matmul(K_D__K_XT, tf.linalg.cholesky_solve(L, K_D__K_TX)) \
                + self.add_diag * tf.eye(tf.cast(tf.shape(K_D__K_Xi)[0], tf.int32))
        epsilon = tf.random.normal((tf.shape(Xi)[0] * self.n_features, self.n_mc_samples))
        chol_Sigma, num_tries = self.try_cholesky(Sigma)
        draws = tf.matmul(chol_Sigma, epsilon) + Mu

        # draws = [x_i] * n_feat, n_mc_samples
        shaped_draws = tf.transpose(tf.reshape(draws, (self.n_features, X_len, self.n_mc_samples)), perm=[2, 1, 0])
        # shaped_draws = n_mc_samples, X, n_feat
        return shaped_draws

    def try_cholesky(self, Sigma):
        try_no = 0
        try:
            try_no += 1
            chol_sigma = tf.linalg.cholesky(Sigma)
        except:
            t_print("Chol ill defined. New diag {}".format(self.add_diag * 11))
            Sigma = Sigma + self.add_diag * 10 * tf.eye(tf.cast(tf.shape(Sigma)[0], tf.int32))
            try:
                try_no += 1
                chol_sigma = tf.linalg.cholesky(Sigma)
            except:
                t_print("Chol ill defined. New diag {}".format(self.add_diag * 111))
                Sigma = Sigma + self.add_diag * 10 * tf.eye(tf.cast(tf.shape(Sigma)[0], tf.int32))
                try:
                    try_no += 1
                    chol_sigma = tf.linalg.cholesky(Sigma)
                except:
                    t_print("Chol ill defined. New diag {}".format(self.add_diag * 1111))
                    Sigma = Sigma + self.add_diag * 10 * tf.eye(tf.cast(tf.shape(Sigma)[0], tf.int32))
                    try_no += 1
                    chol_sigma = tf.linalg.cholesky(Sigma)

        return chol_sigma, try_no
