# -*- coding: utf-8 -*-
from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from datasets.mnist import mnist as MNIST


mnist = MNIST(one_hot=True)


class LeNet(object):
    def __init__(self, c1=6, s1=2, c2=16, s2=2):
        self.model = self.build_structure(c1, s1, c2, s2)
    
    def build_structure(self, c1, s1, c2, s2):
        model = Sequential()
        model.add(Conv2D(c1, 5, activation="relu"))
        model.add(MaxPool2D(pool_size=(s1, s1)))
        model.add(Conv2D(c2, 5, activation="relu"))
        model.add(MaxPool2D(pool_size=(s2, s2)))
        model.add(Flatten())
        model.add(Dense(120, activation="relu"))
        model.add(Dense(84, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])
        return model
    
    def train(self, x, y):
        start = datetime.now()
        self.model.fit(x, y, epochs=1)
        end = datetime.now()
        return (end-start).total_seconds()


if __name__ == "__main__":
    import argparse
    def lenet_params(s):
        try:
            return [int(i) for i in s.split(",")]
        except Exception:
            raise argparse.ArgumentTypeError
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=lenet_params, nargs="+")
    args = parser.parse_args()
    x = mnist.train.images
    y = mnist.train.labels
    x = x.reshape(x.shape[0], 28, 28, 1)
    for param in args.params:
        lenet = LeNet(*param)
        start = datetime.now()
        lenet.train(x, y)
        print(datetime.now()-start)
