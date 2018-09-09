# -*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.linear_model import LinearRegression


def load_data():
	with np.load("mnist.npz") as f:
		x_train, y_train = f['x_train'], f['y_train']
		x_test, y_test = f['x_test'], f['y_test']
	return (x_train.reshape(x_train.shape[0], 28*28), y_train)


def select_column(x, col):
	return x[:,[col]]


def train(x, y):
	regression = LinearRegression()
	start = time.time()
	regression.fit(x,y)
	end = time.time()
	return end-start


if __name__ == "__main__":
	x,y=load_data()
	x=select_column(x, 2)
	print(train(x,y))