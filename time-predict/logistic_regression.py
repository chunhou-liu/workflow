# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

def regression(x, y):
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    return reg


def load_data():
	with np.load("mnist.npz") as f:
		x_train, y_train = f['x_train'], f['y_train']
		x_test, y_test = f['x_test'], f['y_test']
	return (x_train.reshape(x_train.shape[0], 28*28), y_train)


X,Y=load_data()
sizes=[100,200,300,400,500]
counting=[]
for size in sizes:
	reg=LogisticRegression(max_iter=1000,solver="lbfgs")
	x,y=X[:size,:],Y[:size]
	start=datetime.now()
	reg.fit(x,y)
	end=datetime.now()
	print((end-start).total_seconds())
	counting.append((end-start).total_seconds())
time_predict=regression(sizes,counting)
sizes=[3000,4000,5000]
for size in sizes:
	reg=LogisticRegression(max_iter=1000,solver="lbfgs")
	x,y=X[:size,:],Y[:size]
	start=datetime.now()
	reg.fit(x,y)
	end=datetime.now()
	print(time_predict.predict(size),(end-start).total_seconds())