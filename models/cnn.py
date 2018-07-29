# -*- coding: utf-8 -*-
import tensorflow as tf


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

    def train(self, mnist, train_steps=10000, batch_size=100):
        initializer = tf.global_variables_initializer()
        self.session.run(initializer)
        for i in range(train_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            self.session.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})

    def test(self, test_x, test_y):
        return self.session.run(self.accuracy, feed_dict={self.x: test_x, self.y: test_y})


if __name__ == "__main__":
    from datetime import datetime
    from datasets.mnist import mnist
    cout1s = [6, 16, 32, 64, 120]
    cout2s = [32, 64, 120]
    for c1 in cout1s:
        for c2 in cout2s:
            param = (c1, c2)
            cnn = CNNNetwork(*param)
            start = datetime.now()
            cnn.train(mnist, train_steps=2000, batch_size=100)
            end = datetime.now()
            acc = cnn.test(mnist.test.images, mnist.test.labels)
            with open("cnn-log.txt", "a") as f:
                print(param, cnn.X, (end-start).total_seconds(), acc, sep='\t', file=f)
            print(param, cnn.X, (end - start).total_seconds(), acc, sep='\t')
