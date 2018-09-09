# -*- coding: utf-8 -*-
import operator
from datetime import datetime
from sklearn.svm import SVC
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


def rbf_kernel_svc_evaluation(C, gamma, n, d, loops=10):
    total_time_count = 0
    for i in range(loops):
        X, y = dataset_generator(n, d)
        clf = SVC(C=C, gamma=gamma)  # default kernel: RBF
        start = datetime.now()
        clf.fit(X, y)
        end = datetime.now()
        total_time_count += (end-start).total_seconds()
    return total_time_count / loops


if __name__ == "__main__":
    C = [
        0.1,
        1,
        10
    ]
    gamma = [
        0.1,
        0.2,
        0.3,
        0.5
    ]
    for _C in C:
        for _gamma in gamma:
            time = rbf_kernel_svc_evaluation(_C, _gamma, 10000, 2)
            print(_C, _gamma, time)