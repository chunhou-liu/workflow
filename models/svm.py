# -*- coding: utf-8 -*-
import sys
from datetime import datetime
import numpy as np
from sklearn import svm, linear_model
from sklearn.utils import resample
from datasets.mnist import mnist


def log(*arg, **kwarg):
    print(*arg, **kwarg)
    with open("svm-log.txt", "a") as f:
        print(*arg, **kwarg, file=f)


def svm_classifier(X, y):
    clf = svm.SVC(decision_function_shape="ovo")
    start = datetime.now()
    clf.fit(X, y)
    end = datetime.now()
    return clf, (end-start).total_seconds()


def regression(x, y):
    x = np.array([i**2 for i in x]).reshape(-1, 1)
    y = np.array(y)
    model = linear_model.LinearRegression()
    model.fit(x, y)
    log("model coef:", model.coef_, "model intercept:", model.intercept_, sep='\t')
    return model.coef_[0], model.intercept_


assert(all([i.isdigit() for i in sys.argv[1:]]))

sample_numbers = [int(i) for i in sys.argv[1:-3]]
assert(all([i <= len(mnist.train.images) for i in sample_numbers]))

train_times = []
for sample_num in sample_numbers:
    x, y = resample(mnist.train.images, mnist.train.labels, replace=False, n_samples=sample_num)
    model, train_time = svm_classifier(x, y)
    acc = np.sum(model.predict(mnist.test.images) == mnist.test.labels) / len(mnist.test.labels)
    log(sample_num, train_time, acc, sep='\t')
    train_times.append(train_time)

log(sample_numbers)
log(train_times)
log("+"*50)

coef, intercept = regression(sample_numbers, train_times)

test_sample_numbers = [int(i) for i in sys.argv[-3:]]

for sample_num in test_sample_numbers:
    x, y = resample(mnist.train.images, mnist.train.labels, replace=False, n_samples=sample_num)
    model, train_time = svm_classifier(x, y)
    acc = np.sum(model.predict(mnist.test.images) == mnist.test.labels) / len(mnist.test.labels)
    log("prediction time:",coef*sample_num**2 + intercept, "real time:", train_time, "acc:", acc, sep='\t')

log("="*50)