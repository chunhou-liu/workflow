# -*- coding: utf-8 -*-
import logging
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import numpy as np
from datetime import datetime
from datasets.mnist import mnist as MNIST


mnist = MNIST(one_hot=False)


def sample(x: np.ndarray, y: np.ndarray, size: int):
    return resample(x, y, n_samples=size, replace=False)


def svm(x, y, test_x, test_y, size, mark=False):
    if not mark:
        clf = LinearSVC(loss="hinge")
        train_x, train_y = sample(x, y, size)
        start = datetime.now()
        clf.fit(train_x, train_y)
        end = datetime.now()
        acc = np.sum(clf.predict(test_x) == test_y) / len(test_y)
        return (end-start).total_seconds(), acc
    times = []
    accs = []
    for i in range(5):
        clf = LinearSVC(loss="hinge")
        train_x, train_y = sample(x, y, size)
        start = datetime.now()
        clf.fit(train_x, train_y)
        end = datetime.now()
        acc = np.sum(clf.predict(test_x) == test_y) / len(test_y)
        times.append((end-start).total_seconds())
        accs.append(acc)
    return sum(times) / len(times), sum(accs) / len(accs)


def regression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    pred = LinearRegression()
    pred.fit(x, y)
    print(pred.coef_, pred.intercept_)
    return pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, nargs="+")
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--predict", type=int, nargs="+")
    args = parser.parse_args()
    if args.mark:
        result = []
        for size in args.size:
            time, acc = svm(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, size, True)
            result.append((time, acc))
            print(size, time, acc, sep='\t')
        x = [i**2 for i in args.size]
        y = [i[0] for i in result]
        model = regression(x, y)
        preds = [model.predict(i**2) for i in args.predict]
        real = []
        for size in args.predict:
            time, acc = svm(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, size)
            real.append((time, acc))
        print("pred", "real", "acc", sep='\t')
        for i, (j, k) in zip(preds, real):
            print(i, j, k, sep='\t')
    else:
        for size in args.predict:
            time, acc = svm(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, size)
            print(size, time, acc, sep='\t') 


"""
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
            log(sample_num, model.predict(sample_num**2), train_time, acc)
    else:
        for sample_num in args.predict:
            x, y = resample(mnist.train.images, mnist.train.labels, replace=False, n_samples=sample_num)
            train_time, acc = svm_classifier(x, y)
            log(sample_num, train_time, acc)
    log("#"*50)

# python -m models.svm --mark --size 1000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 --predict 2000 17000 47000 35000
"""