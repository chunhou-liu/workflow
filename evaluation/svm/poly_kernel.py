# -*- coding: utf-8 -*-
import operator
from datetime import datetime
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt


def dataset_generator(row, col):
    X = np.random.random(size=[row, col])
    y = np.random.randint(0, 10, size=(row,))
    return X, y


def poly_kernel_svc_evaluation(n, d, degree, loops=10):
    total_time_count = 0
    for i in range(loops):
        X, y = dataset_generator(n, d)
        clf = SVC(kernel="poly", degree=degree)  # ploynomial kernel
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


if __name__ == "__main__":
    params = [
        (5000, 10, 3),
        (5000, 10, 4),
        (5000, 10, 5),
        (5000, 10, 6),
        (5000, 10, 7),
        (5000, 10, 8),
        (5000, 10, 9),
        (5000, 10, 10),
    ]
    test_params = [
        (12000, 10, 11),
        (15000, 10, 12)
    ]
    X,y=[],[]
    for n,d,degree in params:
        X.append(degree)
        y_ = poly_kernel_svc_evaluation(n,d,degree)
        print(degree, y_)
        y.append(y_)
    reg = poly_kernel_linear_regression(X, y)
    for n,d,degree in test_params:
        X.append(degree)
        y_ = poly_kernel_svc_evaluation(n,d,degree)
        print(degree, y_)
        y.append(y_)
        print(y_, reg.predict(degree))
    points = sorted(zip(X,y),key=operator.itemgetter(0))
    plt.plot([point[0] for point in points], [point[1] for point in points])
    plt.show()