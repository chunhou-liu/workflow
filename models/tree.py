# -*- coding: utf-8 -*-
import sys
from datetime import datetime
import numpy as np
from sklearn import tree, linear_model
from datasets.mnist import mnist


def log(*args, **kwargs):
    print(*args, **kwargs)
    with open("tree-log.txt", "a") as f:
        print(*args, **kwargs, file=f)


def cartree_classifier(max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    start = datetime.now()
    clf.fit(mnist.train.images, mnist.train.labels)
    end = datetime.now()
    return clf, (end-start).total_seconds()


def regression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    log(reg.coef_, reg.intercept_)
    return reg


assert(all([i.isdigit() for i in sys.argv[1:]]))
train_depth = [int(i) for i in sys.argv[1:-3]]
train_times = []
for max_depth in train_depth:
    model, train_time = cartree_classifier(max_depth)
    train_times.append(train_time)
    acc = np.sum(model.predict(mnist.test.images) == mnist.test.labels) / len(mnist.test.labels)
    log(max_depth, train_time, acc, sep='\t')
log(train_depth)
log(train_times)
log('$'*50)

reg = regression(train_depth, train_times)
test_depth = [int(i) for i in sys.argv[-3:]]
for max_depth in test_depth:
    model, train_time = cartree_classifier(max_depth)
    train_times.append(train_time)
    acc = np.sum(model.predict(mnist.test.images) == mnist.test.labels) / len(mnist.test.labels)
    log(max_depth, "prediction time:", reg.predict(max_depth), "real time: ", train_time, acc, sep='\t')
log('@'*50)