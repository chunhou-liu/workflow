# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.linear_model import LinearRegression


def load_data():
	with np.load("mnist.npz") as f:
		x_train, y_train = f['x_train'], f['y_train']
		x_test, y_test = f['x_test'], f['y_test']
	return (x_train.reshape(x_train.shape[0], 28*28), to_categorical(y_train,num_classes=10))


def create_predict_function(x,y):
	x=np.array(x).reshape(-1,1)
	y=np.array(y)
	reg=LinearRegression()
	reg.fit(x,y)
	return reg

def get_training_model():
	model = Sequential()
	model.add(Dense(32, activation="relu",input_dim=784))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation="relu"))	
	model.add(Dropout(0.5))
	model.add(Dense(10, activation="softmax"))

	sgd=SGD(lr=0.05,decay=1e-6,momentum=0.9,nesterov=False)
	model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
	return model

X,Y=load_data()
sizes=[]
counting=[]
for size in range(1000,2000,50):
	x,y=X[:size,:],Y[:size,:]
	model=get_training_model()
	start=datetime.now()
	model.fit(x,y,epochs=20,batch_size=50)
	end=datetime.now()
	sizes.append(size*20/50)
	counting.append((end-start).total_seconds())
predictor=create_predict_function(sizes,counting)
predictions=predictor.predict(X.shape[0]*20/50)
model=get_training_model()
start=datetime.now()
model.fit(X,Y,epochs=20,batch_size=50)
end=datetime.now()
print(predictions,(end-start).total_seconds())