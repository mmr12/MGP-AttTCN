import numpy as np
import tensorflow as tf

def GP_loss(model, inputs, labels, weighted_loss=None):
    out = model(inputs)
    temp_labels = tf.reshape(labels, (-1, 1))
    y_star = tf.concat([temp_labels, tf.ones_like(temp_labels) - temp_labels], axis=1, name="y_star")
    if weighted_loss is not None:
        float_y_star = tf.cast(y_star, dtype=tf.float32)
        weights = float_y_star[:, 0] / weighted_loss + float_y_star[:, 1]
    else:
        weights = tf.convert_to_tensor([1], dtype=tf.float32)
    return tf.compat.v1.losses.softmax_cross_entropy(y_star, out, weights=weights)


def grad(model, inputs, targets, ratio_weights=None, multi_class=True, GP=False, weighted_loss=None):
    if GP == True:
        with tf.GradientTape() as tape:
            loss_value = GP_loss(model, inputs, targets, weighted_loss)
    else:
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, ratio_weights, multi_class=multi_class)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
