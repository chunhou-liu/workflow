# -*- coding: utf-8 -*-
import tensorflow as tf
from models import lib


class LeNet(object):
    def __init__(self, c1, c3, c5):
        self.var = 0
        self.x = tf.placeholder(tf.float32, shape=(None, 784))
        self.y = tf.placeholder(tf.int64, shape=(None,))
        self.lenet = self.install_network(c1, c3, c5)
        self.train_step, self.prediction_step, self.accuracy_step = self.install_model()
        self.sess = tf.Session()
        print(self.var)

    @staticmethod
    def fully_connected(input, units):
        weight = tf.Variable(tf.truncated_normal([input.shape.as_list()[1], units], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[units]))
        return tf.matmul(input, weight) + bias

    def install_network(self, c1, c3, c5):
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(input_layer, c1, 5, activation=tf.nn.relu)
        self.var += conv1.shape.as_list()[1] ** 2 * 1 * c1
        scale2 = lib.max_pool_2x2(conv1)
        conv3 = tf.layers.conv2d(scale2, c3, 5, activation=tf.nn.relu)
        self.var += conv3.shape.as_list()[1] ** 2 * c1 * c3
        scale4 = lib.max_pool_2x2(conv3)
        conv5 = tf.layers.conv2d(scale4, c5, 5, activation=tf.nn.relu)
        self.var += conv5.shape.as_list()[1] ** 2 * c3 * c5
        flatten = tf.layers.flatten(conv5)
        fc1 = self.fully_connected(flatten, 84)
        output = self.fully_connected(fc1, 10)
        return output

    def install_model(self):
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.lenet))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        prediction = tf.argmax(self.lenet, 1)
        correct_prediction = tf.cast(tf.equal(prediction, self.y), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        return train_step, prediction, accuracy

    def train(self, mnist, train_steps=10000, batch_size=100):
        initializer = tf.global_variables_initializer()
        self.sess.run(initializer)
        for i in range(train_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})

    def test(self, test_x, test_y):
        return self.sess.run(self.accuracy_step, feed_dict={self.x: test_x, self.y: test_y})


if __name__ == "__main__":
    from datasets.mnist import mnist
    params = [(i, j, k) for i in [6, 16, 32, 64] for j in [16, 32, 64] for k in [16, 32, 64, 120]]
    for param in params:
        lenet = LeNet(*param)
        print(lenet.var)
    lenet.train(mnist, train_steps=1000)
    print(lenet.test(mnist.test.images, mnist.test.labels))
