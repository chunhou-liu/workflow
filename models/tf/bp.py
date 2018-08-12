# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn import linear_model
from datasets.mnist import mnist as MNIST


mnist = MNIST(True)


def _weight_variable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


def _bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def _bp_network(units: int):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
    y = tf.placeholder(dtype=tf.int64, shape=(None, 10))
    weight = _weight_variable([784, units])
    bias = _bias_variable([units])
    hide = tf.sigmoid(tf.matmul(x, weight) + bias)
    weight = _weight_variable([units, 10])
    bias = _bias_variable([10])
    output = tf.nn.softmax(tf.matmul(hide, weight) + bias)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    accuracy = tf.equal (tf.argmax (y, 1), tf.argmax (output, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    return x, y, train, accuracy


def bp_classifier(units, train_steps=10000, batch_size=100):
    x, y, train, accuracy = _bp_network(units)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = datetime.now()
    for i in range(train_steps):
        batch_x, batch_y  = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y: batch_y})
    end = datetime.now()
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    return (end-start).total_seconds(), acc


def log(*args, **kwargs):
    print(*args, **kwargs, sep='\t')
    with open("bp-log.txt", "a") as f:
        print(*args, **kwargs, file=f, sep='\t')


def regression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    log(reg.coef_, reg.intercept_)
    return reg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", type=int, default=[100])
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--predict", type=int, nargs="+", default=[300])
    parser.add_argument("--mark", action="store_true")
    args = parser.parse_args()
    if args.mark:
        log("units", "train time", "acc")
        train_times = []
        for unit in args.nodes:
            train_time, acc = bp_classifier(unit, train_steps=args.train_steps, batch_size=args.batch_size)
            log(unit, train_time, acc)
            train_times.append(train_time)
        log("%"*50)
        model = regression(args.nodes, train_times)
        log("units", "train time", "pred time", "acc")
        for unit in args.predict:
            train_time, acc = bp_classifier(unit, train_steps=args.train_steps, batch_size=args.batch_size)
            log(unit, train_time, model.predict(unit), acc)
    else:
        log("units", "train time", "acc")
        for unit in args.predict:
            train_time, acc = bp_classifier(unit, train_steps=args.train_steps, batch_size=args.batch_size)
            log(unit, train_time,acc)
    log("#"*50)

# python -m models.bp --nodes 50 100 200 300 400 500 800 1000 1300 1500 1800 2000 --train_steps 10000 --predict 600 2200 --mark