# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model


def svm_regression(x, y):
    x = np.array([i**2 for i in x]).reshape(-1, 1)
    y = np.array(y)
    model = linear_model.LinearRegression()
    model.fit(x, y)
    print(model.coef_, model.intercept_, sep='\t')
    return model.coef_[0], model.intercept_
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    y = [1.330465, 3.960027, 7.258625, 11.521135, 16.483816, 21.687696, 27.379265, 33.594721, 41.229299, 47.924979]
    plt.plot(x, y)
    plt.show()
    svm_regression(x, y)