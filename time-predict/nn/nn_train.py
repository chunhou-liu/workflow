# -*- coding: utf-8 -*-
from datetime import datetime
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import os
from lstm import build_model


(X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.abspath("../mnist.npz"))
# reshape to be [samples][pixels][width][height]
# bpnet only
#X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32')
# cnn net
X_train = X_train.reshape(X_train.shape[0], 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--dataset", nargs="+", type=int)
args=parser.parse_args()
Y=[]
for size in args.dataset:
    model = build_model()
    start=datetime.now()
    model.fit(X_train[:size],y_train[:size], batch_size=20, epochs=1,verbose=0)
    end=datetime.now()
    print(size, (end-start).total_seconds())
    Y.append((end-start).total_seconds())
print(args.dataset, Y, sep='\n')