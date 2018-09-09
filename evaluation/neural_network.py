# -*- coding: utf-8 -*-
import operator
from datetime import datetime
import tensorflow as tf
from matplotlib import pyplot as plt
from datasets.mnist import mnist as MNIST


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
        fc1 = self.fully_connected(flatten, 1024)
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


def cnn_classifier_evaluation():
    params = [
        (1000, 50),
        (2000, 50),
        (3000, 50),
        (4000, 50),
        (5000, 50),
        (10000, 50),
        (20000, 50)
    ]
    X,y=[],[]
    for train_steps, batch_size in params:
        cnn_network = CNNNetwork(32, 64)
        train_time, acc = cnn_network.train(train_steps, batch_size)
        X.append(train_steps*batch_size)
        y.append(train_time)
        print(acc, train_steps*batch_size, train_time)
    points = sorted(zip(X,y),key=operator.itemgetter(0))
    plt.plot([point[0] for point in points], [point[1] for point in points])
    plt.show()


def conv2d_evaluation()


if __name__ == "__main__":
    cnn_classifier_evaluation()