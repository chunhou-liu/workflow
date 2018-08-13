# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime
import tensorflow as tf


def evaluate_cnn(M, K, Cin, Cout):
    X = np.random.random(size=(50000, M, M, Cin))
    sess = tf.Session()
    input = tf.placeholder(dtype=tf.float32, shape=(None, M, M, Cin))
    filters = tf.Variable(tf.truncated_normal(shape=[K,K,Cin,Cout]))
    strides = [1,1,1,1]
    padding="VALID"
    conv = tf.nn.conv2d(input, filters, strides, padding)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = datetime.now()
        for i in range(100):
            sess.run(conv, feed_dict={input: X[i*50:(i+1)*50]})
        end = datetime.now()
    return (end-start).total_seconds() / 100


def evaluate_dense(units, features):
    X = np.random.random(size=(50000, features))
    sess = tf.Session()
    input = tf.placeholder(dtype=tf.float32, shape=(None, features))
    dense = tf.layers.dense(input, units)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = datetime.now()
        for i in range(1000):
            sess.run(dense, feed_dict={input: X[i*50:(i+1)*50]})
        end = datetime.now()
    return (end-start).total_seconds() / 1000


    
"""
if __name__ == "__main__":
    import operator
    from matplotlib import pyplot as plt
    params = [
        (28,5,3,6),
        (28,3,3,6),
        (28,7,3,6),
        (28,3,5,6),
        (28,3,1,6),
        (28,5,7,9)
    ]
    times = []
    for param in params:
        times.append(evaluate_cnn(*param))
    X = []
    for M, K, Cin, Cout in params:
        X.append(M**2*K**2*Cin*Cout)
    data = sorted(zip(times, X), key=operator.itemgetter(1))
    plt.plot([i[1] for i in data], [i[0] for i in data])
    plt.show()
"""

if __name__ == "__main__":
    import operator
    from matplotlib import pyplot as plt
    params = [
        (100, 10),
        (200, 10),
        (300, 10),
        (500, 10),
        (700, 10),
        (900, 10),
        (1000, 10),
        (1500, 10)
    ]
    times = []
    for param in params:
        times.append(evaluate_dense(*param))
    X = []
    for units, features in params:
        X.append(units)
    data = sorted(zip(times, X), key=operator.itemgetter(1))
    plt.plot([i[1] for i in data], [i[0] for i in data])
    plt.show()