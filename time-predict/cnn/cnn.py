# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import numpy as np
from sklearn import linear_model
import tensorflow as tf
from mnist import mnist as MNIST


mnist = MNIST()


class CNNNetwork(object):
    def __init__(self, cout1, cout2, features=784, size=28, classes=10):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, features))
        self.y = tf.placeholder(dtype=tf.int64, shape=(None, ))
        self.classes = classes
        self.session = tf.Session()
        self.cnn, self.X = self.build_network(cout1, cout2, size, classes)
        self.train_step, self.prediction, self.accuracy = self.build_model()

    @staticmethod
    def fully_connected(input, units):
        weight = tf.Variable(tf.truncated_normal([input.shape.as_list()[1], units], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[units]))
        return tf.matmul(input, weight) + bias

    def build_network(self, cout1, cout2, size, classes):
        input_layer = tf.reshape(self.x, shape=[-1, size, size, 1])
        conv1 = tf.layers.conv2d(input_layer, cout1, 5, activation=tf.nn.relu)
        
        sample1 = tf.layers.average_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(sample1, cout2, 5, activation=tf.nn.relu)
        
        sample2 = tf.layers.average_pooling2d(conv2, 2, 2)
        flatten = tf.layers.flatten(sample2)
        fc1 = self.fully_connected(flatten, 200)
        fc1 = tf.layers.dropout(fc1, 0.25)
        output_layer = self.fully_connected(fc1, classes)
        x = conv1.shape.as_list()[1] * conv1.shape.as_list()[2] * 25 * 1 * cout1
        x += conv2.shape.as_list()[1] * conv2.shape.as_list()[2] * 25 * cout1 * cout2
        return output_layer, x

    def build_model(self):
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.cnn))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        prediction = tf.argmax(self.cnn, 1)
        correct_prediction = tf.cast(tf.equal(prediction, self.y), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        return train_step, prediction, accuracy

    def train(self, train_steps=10000, batch_size=100):
        initializer = tf.global_variables_initializer()
        self.session.run(initializer)
        start = datetime.now()
        for i in range(train_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            self.session.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})
        end = datetime.now()
        acc = self.session.run(self.accuracy, feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels})
        return (end-start).total_seconds(), acc

    def test(self, test_x, test_y):
        return self.session.run(self.accuracy, feed_dict={self.x: test_x, self.y: test_y})


def log(*args, **kwargs):
    print(*args, **kwargs, sep='\t')
    with open("cnn-log.txt", "a") as f:
        print(*args, **kwargs, file=f, sep='\t')


def regression(x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    log(reg.coef_, reg.intercept_)
    return reg


def cnn_params(s):
    try:
        return tuple(map(int, s.split(",")))
    except:
        raise argparse.ArgumentTypeError("must be a tuple of integers")


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("-c", "--conv", nargs="+", default=[(32, 64)], type=cnn_params, help="conv layer nodes")
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--predict", nargs="+", type=cnn_params, default=[(32, 64)])
    args = parser.parse_args()

    if args.mark:
        x = []
        train_times = []
        for param in args.conv:
            cnn = CNNNetwork(*param)
            train_time, acc = cnn.train(train_steps=args.train_steps, batch_size=args.batch_size)
            x.append(cnn.X)
            train_times.append(train_time)
            log(param, cnn.X, train_time, acc)
        log("%"*50)
        model = regression(x, train_times)
        for param in args.predict:
            cnn = CNNNetwork(*param)
            train_time, acc = cnn.train(train_steps=args.train_steps, batch_size=args.batch_size)
            log(param, cnn.X, model.predict(cnn.X), train_time, acc)
    else:
        for param in args.predict:
            cnn = CNNNetwork(*param)
            train_time, acc = cnn.train(train_steps=args.train_steps, batch_size=args.batch_size)
            log(param, cnn.X, train_time, acc)
    log("#"*50)


# python -m models.cnn --conv 16,32 16,64 16,120 32,64 32,120 64,120 --predict 30,50 --mark
"""
if __name__ == "__main__":
    train_steps=[100,200,300,400,500]
    batch_size=50
    sizes=[]
    counting=[]
    for step in train_steps:
        cnn = CNNNetwork(16,32)
        start=datetime.now()
        cnn.train(step,batch_size)
        end=datetime.now()
        sizes.append(step*batch_size)
        print(step*batch_size,(end-start).total_seconds())
        counting.append((end-start).total_seconds())
    time_predict=regression(sizes,counting)
    train_steps=[1000,2000,3000]
    batch_size=50
    for train_step in train_steps:
        #time_predict.predict(train_step*batch_size)
        print(time_predict.predict(train_step*batch_size))
        cnn = CNNNetwork(16,32)
        start=datetime.now()
        cnn.train(train_step,batch_size)
        end=datetime.now()
        print(train_step*batch_size,(end-start).total_seconds())