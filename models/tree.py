# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import numpy as np
from sklearn import tree, linear_model
from datasets.mnist import mnist as MNIST


mnist = MNIST(False)


def log(*args, **kwargs):
    print(*args, **kwargs, sep='\t')
    with open("tree-log.txt", "a") as f:
        print(*args, **kwargs, file=f, sep='\t')


def cartree_classifier(max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    start = datetime.now()
    clf.fit(mnist.train.images, mnist.train.labels)
    end = datetime.now()
    acc = np.sum(clf.predict(mnist.test.images) == mnist.test.labels) / len(mnist.test.labels)
    return (end-start).total_seconds(), acc


def regression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    log(reg.coef_, reg.intercept_)
    return reg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", nargs="+", type=int, default=[10, 20, 30, 40, 70, 100, 150, 200, 500, 1000])
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--predict", nargs="+", type=int, default=[45, 60, 300, 700])
    args = parser.parse_args()
    if args.mark:
        train_times = []
        for max_depth in args.depth:
            train_time, acc = cartree_classifier(max_depth)
            train_times.append(train_time)
            log(max_depth, train_time, acc)
        log('$'*50)
        model = regression(args.depth, train_times)
        for max_depth in args.predict:
            train_time, acc = cartree_classifier(max_depth)
            train_times.append(train_time)
            log(max_depth, model.predict(max_depth), train_time, acc)
    else:
        for max_depth in args.predict:
            train_time, acc = cartree_classifier(max_depth)
            log(max_depth, train_time, acc)
    log("="*50)


# python -m models.tree --mark