# -*- coding: utf-8 -*-
import argparse
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf


def evaluate_dense(units):
    data = np.random.random(size=(20000, 784))
    x=tf.placeholder(tf.float32, (None, 784))
    nn = tf.layers.dense(x, units, tf.nn.relu)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = datetime.now()
    loops = 100
    for i in range(loops):
        sess.run(nn, feed_dict={x: data})
    end = datetime.now()
    return (end-start).total_seconds() / loops


def evaluate_convolve(kernel_size, in_channels, out_channels):
    data = np.random.random(size=(20000, 784*in_channels))
    x=tf.placeholder(tf.float32, shape=(None, 784*in_channels))
    y=tf.reshape(x, [-1, 28, 28, in_channels])
    conv=tf.nn.conv2d(y, tf.Variable(tf.truncated_normal([kernel_size, kernel_size, in_channels, out_channels])), padding="SAME", strides=[1,1,1,1])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = datetime.now()
    loops = 100
    for i in range(loops):
        sess.run(conv, feed_dict={x: data})
    end = datetime.now()
    return (end-start).total_seconds() / loops


def cnn_params(s):
    try:
        return [int(i) for i in s.split(",")]
    except argparse.ArgumentTypeError:
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=int, nargs="+", default=[100, 200, 3000, 10000])
    parser.add_argument("--convolve", type=cnn_params, nargs="+", default=[(5,32,64)])
    parser.add_argument("--evaluate", type=str, choices=["dense", "convolve"])
    args = parser.parse_args()
    if args.evaluate == "dense":
        for units in args.dense:
            print(units, evaluate_dense(units), sep='\t')
    elif args.evaluate == "convolve":
        for kernel_size, in_channels, out_channels in args.convolve:
            print(kernel_size, in_channels, out_channels, evaluate_convolve(kernel_size, in_channels, out_channels))
