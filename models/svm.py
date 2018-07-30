# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import numpy as np
from sklearn import svm, linear_model
from sklearn.utils import resample
from datasets.mnist import mnist as MNIST


mnist = MNIST(False)


def log(*arg, **kwarg):
    print(*arg, **kwarg, sep='\t')
    with open("svm-log.txt", "a") as f:
        print(*arg, **kwarg, file=f, sep='\t')


def svm_classifier(X, y):
    clf = svm.SVC(decision_function_shape="ovo")
    start = datetime.now()
    clf.fit(X, y)
    end = datetime.now()
    acc = np.sum(clf.predict(mnist.test.images) == mnist.test.labels) / len(mnist.test.labels)
    return (end-start).total_seconds(), acc


def regression(x, y):
    x = np.array([i**2 for i in x]).reshape(-1, 1)
    y = np.array(y)
    model = linear_model.LinearRegression()
    model.fit(x, y)
    log(model.coef_, model.intercept_)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--size", type=int, nargs="+", default=[1000])
    parser.add_argument("--predict", type=int, nargs="+", default=[2000])
    args = parser.parse_args()
    if args.mark:
        train_times = []
        for sample_num in args.size:
            x, y = resample(mnist.train.images, mnist.train.labels, replace=False, n_samples=sample_num)
            train_time, acc = svm_classifier(x, y)
            train_times.append(train_time)
            log(sample_num, train_time, acc)
        log("%"*50)
        model = regression(args.size, train_times)
        for sample_num in args.predict:
            x, y = resample(mnist.train.images, mnist.train.labels, replace=False, n_samples=sample_num)
            train_time, acc = svm_classifier(x, y)
            train_times.append(train_time)
            log(sample_num, model.predict(sample_num**2), train_time, acc)
    else:
        for sample_num in args.predict:
            x, y = resample(mnist.train.images, mnist.train.labels, replace=False, n_samples=sample_num)
            train_time, acc = svm_classifier(x, y)
            train_times.append(train_time)
            log(sample_num, train_time, acc)
    log("#"*50)

# python -m models.svm --mark --size 1000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 --predict 2000 17000 47000 35000