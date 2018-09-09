# -*- coding: utf-8 -*-
import operator
from datetime import datetime
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt


"""
    once d is settled, the time complexity is:
        SVC rbf kernel O(n^2)
        SVC poly kernel O(n^2)
        LinearSVC O(n)
"""

def dataset_generator(row, col):
    X = np.random.random(size=[row, col])
    y = np.random.randint(0, 10, size=(row,))
    return X, y


def rbf_kernel_svc_evaluation(n, d, loops=10):
    total_time_count = 0
    for i in range(loops):
        X, y = dataset_generator(n, d)
        clf = SVC(C=0.1, gamma=0.1)  # default kernel: RBF
        start = datetime.now()
        clf.fit(X, y)
        end = datetime.now()
        total_time_count += (end-start).total_seconds()
    return total_time_count / loops


def rbf_kernel_linear_regression(X, y):
    X_ = np.array(X, dtype=np.float64).reshape(-1,1)
    y_ = np.array(y)
    reg = LinearRegression()
    reg.fit(X_, y_)
    return reg


def poly_kernel_svc_evaluation(n, d, loops=10):
    total_time_count = 0
    for i in range(loops):
        X, y = dataset_generator(n, d)
        clf = SVC(kernel="poly")  # ploynomial kernel
        start = datetime.now()
        clf.fit(X, y)
        end = datetime.now()
        total_time_count += (end-start).total_seconds()
    return total_time_count / loops


def poly_kernel_linear_regression(X, y):
    X_ = np.array(X, dtype=np.float64).reshape(-1,1)
    y_ = np.array(y)
    reg = LinearRegression()
    reg.fit(X_, y_)
    return reg


def linear_kernel_svc_evaluation(n,d,loops=10):
    total_time_count = 0
    for i in range(loops):
        X, y = dataset_generator(n, d)
        clf = LinearSVC("l2")  # default kernel: RBF
        start = datetime.now()
        clf.fit(X, y)
        end = datetime.now()
        total_time_count += (end-start).total_seconds()
    return total_time_count / loops


def linear_kernel_linear_regression(X, y):
    X_ = np.array(X, dtype=np.float64).reshape(-1,1)
    y_ = np.array(y)
    reg = LinearRegression()
    reg.fit(X_, y_)
    return reg


if __name__ == "__main__":
    params = [
        (1000, 10),
        (2000, 10),
        (3000, 10),
        (4000, 10),
        (5000, 10),
        (7000, 10),
        (8000, 10),
        (10000, 10),
    ]
    test_params = [
        (12000, 10),
        (15000, 10)
    ]
    X,y=[],[]
    for n,d in params:
        X.append(n)
        y_ = linear_kernel_svc_evaluation(n,d)
        #y_ = rbf_kernel_svc_evaluation(n,d)
        print(n, y_)
        y.append(y_)
    reg = linear_kernel_linear_regression(X, y)
    for n,d in test_params:
        X.append(n)
        y_ = linear_kernel_svc_evaluation(n,d)
        print(n, y_)
        y.append(y_)
        print(y_, reg.predict(n))
    points = sorted(zip(X,y),key=operator.itemgetter(0))
    plt.plot([point[0] for point in points], [point[1] for point in points])
    plt.show()
