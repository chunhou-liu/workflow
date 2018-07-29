# -*- coding: utf-8 -*-
import tensorflow as tf


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape), dtype=tf.float32)


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")