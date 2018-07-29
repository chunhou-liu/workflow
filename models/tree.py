# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np
from sklearn import tree
from datasets.mnist import mnist


class CARTree(object):
    def __init__(self, max_depth):
        self.clf = tree.DecisionTreeClassifier(max_depth=max_depth)

    def train(self):
        self.clf.fit(mnist.train.images, mnist.train.labels)

    def test(self):
        return np.sum(self.clf.predict(mnist.test.images) == mnist.test.labels) / len(mnist.test.labels)


if __name__ == "__main__":
    depth = list(range(10, 110, 10))
    y = []
    for max_depth in depth:
        cart = CARTree(max_depth)
        start = datetime.now()
        cart.train()
        end = datetime.now()
        print(max_depth, (end-start).total_seconds(), sep='\t')
        y.append((end-start).total_seconds())
    with open("tree-log.txt", "w") as f:
        print(depth, file=f)
        print(y, file=f)
    print(depth)
    print(y)