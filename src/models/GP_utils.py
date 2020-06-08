#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import tensorflow as tf

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)


# reformatted from M Moor and J Futoma

def kroneker_matrix(M, idx1, idx2=None): #y
    if idx2 is None:
        idx2 = idx1
    grid = tf.meshgrid(idx1, idx2)
    idx = tf.stack((grid[0], grid[1]), -1)
    return tf.gather_nd(M, idx)


def OU_kernel(length, x1, x2):
    x1 = tf.reshape(x1, [-1, 1])  # colvec
    x2 = tf.reshape(x2, [1, -1])  # rowvec
    K = tf.exp(-tf.abs(x1 - x2) / length)
    return K


def K_vitals_initialiser(shape, partition_info=None, dtype=None):
    # initialise lengths to be 0.01 for vitals and 5 for blood tests
    output = np.ones(shape[0])
    output[7:] = 0.01
    return tf.diag(tf.convert_to_tensor(output, dtype=tf.float32))


def K_labs_initialiser(shape, partition_info=None, dtype=None):
    # initialise lengths to be 0.01 for vitals and 5 for blood tests
    output = np.ones(shape[0])
    output[:7] = 0.01
    return tf.diag(tf.convert_to_tensor(output, dtype=tf.float32))
