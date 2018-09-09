# -*- coding: utf-8 -*-
import time
from datetime import datetime
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


def load_data():
	with np.load("mnist.npz") as f:
		x_train, y_train = f['x_train'], f['y_train']
		x_test, y_test = f['x_test'], f['y_test']
	return (x_train.reshape(x_train.shape[0], 28*28), y_train)


def select_column(x, col):
	return x[:,[col]]


def select_part(x, size):
	if len(x.shape) > 1:
		return x[:size, :]
	return x[:size]

def train(x, y):
	regression = DecisionTreeClassifier()
	start = time.time()
	regression.fit(x,y)
	end = time.time()
	return end-start


def MODELING(TRAIN_TIME,SIZES):
	regression=LinearRegression()
	TRAIN_TIME=np.array(TRAIN_TIME)
	SIZES=np.array(SIZES).reshape(-1,1)
	regression.fit(SIZES,TRAIN_TIME)
	return regression


def MODELING2(TRAIN_TIME,SIZES):
	regression=LinearRegression()
	TRAIN_TIME=np.array(TRAIN_TIME)
	SIZES=np.array([i**2 for i in SIZES]).reshape(-1,1)
	regression.fit(SIZES,TRAIN_TIME)
	return regression

if __name__ == "__main__":
	X,Y=load_data()
	sizes=list(range(1000, 50000,5000))
	TEST_SIZES=list(range(50000,61000,1000))
	EXPERIMENT_RESULT=[]
	for size in sizes:
		x=select_part(X, size)
		y=select_part(Y, size)
		TRAIN_TIME=train(x,y)
		print(TRAIN_TIME)
		EXPERIMENT_RESULT.append(TRAIN_TIME)
	MODEL=MODELING(EXPERIMENT_RESULT,sizes)
	MODEL2=MODELING2(EXPERIMENT_RESULT,sizes)
	PREDICT_RESULT=[]
	for size in TEST_SIZES:
		PREDICT=MODEL.predict(size)[0]
		PREDICT2=MODEL2.predict(size**2)[0]
		x=select_part(X, size)
		y=select_part(Y, size)
		TRAIN_TIME=train(x,y)
		print(size,PREDICT,PREDICT2,TRAIN_TIME)
		PREDICT_RESULT.append((PREDICT,PREDICT2,TRAIN_TIME))