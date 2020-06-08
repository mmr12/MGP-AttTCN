import os
import sys

import tensorflow as tf
from tensorflow import keras

cwd = os.path.dirname(os.path.abspath(__file__))
head = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
sys.path.append(head)


def make_model(time_window, no_channels, L2reg, DO, num_layers, kernel_size=2, stride=1, add_classification_layer=True,
               filters_per_layer=None):
    no_initial_channels = no_channels
    if filters_per_layer is None:
        no_channels = no_initial_channels
    else:
        no_channels = filters_per_layer
        # residual block
    layers = [keras.layers.Conv1D(filters=no_channels, kernel_size=kernel_size, strides=1, padding='causal',
                                  dilation_rate=1, activation=tf.nn.relu,
                                  input_shape=(time_window, no_initial_channels),
                                  kernel_regularizer=keras.regularizers.l2(L2reg[0]),
                                  name="conv00"),
              keras.layers.Dropout(DO[0],
                                   name="DropOut00"),
              keras.layers.Conv1D(filters=no_channels, kernel_size=kernel_size, strides=1, padding='causal',
                                  dilation_rate=1, activation=tf.nn.relu,
                                  kernel_regularizer=keras.regularizers.l2(L2reg[0]),
                                  name="conv01"),
              keras.layers.Dropout(DO[0],
                                   name="DropOut01")]
    for i in range(1, num_layers):
        layers += [keras.layers.Conv1D(filters=no_channels, kernel_size=kernel_size, strides=1, padding='causal',
                                       dilation_rate=2 ** i, activation=tf.nn.relu,
                                       kernel_regularizer=keras.regularizers.l2(L2reg[i]),
                                       name="conv{}0".format(i)),
                   keras.layers.Dropout(DO[i],
                                        name="DropOut{}0".format(i)),
                   keras.layers.Conv1D(filters=no_channels, kernel_size=kernel_size, strides=1, padding='causal',
                                       dilation_rate=2 ** i, activation=tf.nn.relu,
                                       kernel_regularizer=keras.regularizers.l2(L2reg[i]),
                                       name="conv{}1".format(i)),
                   keras.layers.Dropout(DO[i],
                                        name="DropOut{}1".format(i))]
    if add_classification_layer:
        layers.append(LastDimDenseLayer(no_channels, 2))
    model = keras.Sequential(layers)
    return model


class LastDimDenseLayer(tf.keras.layers.Layer):
    def __init__(self, no_channels, num_outputs):
        super(LastDimDenseLayer, self).__init__()
        self.kernel = self.add_variable("LastTimestepDense",
                                        shape=[no_channels, num_outputs])

    def build(self, input_shape):
        pass

    def call(self, input):
        return tf.matmul(input[:, -1, :], self.kernel)
