import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

# appending head path
cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir, os.pardir))
sys.path.append(head)

from src.models.TCN import make_model


class AttTCN_beta:
    def __init__(self,
                 time_window,
                 n_channels,
                 num_layers,
                 DO,
                 L2reg,
                 kernel_size=2,
                 stride=1,
                 sigmoid_beta=False):
        self.betaTCN = make_model(time_window=time_window,
                                  no_channels=n_channels,
                                  L2reg=L2reg,
                                  DO=DO,
                                  num_layers=num_layers,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  add_classification_layer=False)

        self.beta_layer_pos = keras.layers.Dense(n_channels, input_shape=[n_channels], name="beta_pos_weights")
        self.beta_layer_neg = keras.layers.Dense(n_channels, input_shape=[n_channels], name="beta_neg_weights")

        self.trainable_variables =self.betaTCN.trainable_variables + \
                                   self.beta_layer_pos.trainable_variables + \
                                   self.beta_layer_neg.trainable_variables
        self.num_layers = num_layers
        self.sigmoid_beta = sigmoid_beta

    def __call__(self, inputs):
        # Note that the activation on alpha and the output are only valid if for a model trained on the last timestep
        if self.sigmoid_beta:
            beta_pos = tf.expand_dims(keras.activations.sigmoid(self.beta_layer_pos(self.betaTCN(inputs))), -1)
            beta_neg = tf.expand_dims(keras.activations.sigmoid(self.beta_layer_neg(self.betaTCN(inputs))), -1)
        else:
            beta_pos = tf.expand_dims(self.beta_layer_pos(self.betaTCN(inputs)), -1)
            beta_neg = tf.expand_dims(self.beta_layer_neg(self.betaTCN(inputs)), -1)
        _ = self.get_weights()
        self.beta = tf.concat([beta_pos, beta_neg], -1)
        expanded_inputs = tf.broadcast_to(tf.expand_dims(inputs, -1), list(self.beta.shape))

        return tf.reduce_sum(self.beta * expanded_inputs, [1, 2])

    def get_weights(self):
        self.trainable_variables = self.betaTCN.trainable_variables + \
                                   self.beta_layer_pos.trainable_variables + \
                                   self.beta_layer_neg.trainable_variables
        return self.trainable_variables

    def set_weights(self, weights):
        if not isinstance(weights[0], np.ndarray):
            weights = [weights[i].numpy() for i in range(len(weights))]
        start = 0
        end = self.num_layers * 4
        self.betaTCN.set_weights(weights[start: end])
        start = end
        end += 2
        self.beta_layer_pos.set_weights(weights[start: end])
        start = end
        end += 2
        self.beta_layer_neg.set_weights(weights[start: end])
