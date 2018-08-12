# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.datasets import mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data("D:\\workspace\\workflow\\models\\mnist.npz")
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


def cnn_param(s):
    try:
        return (int(i) for i in s.split(","))
    except argparse.ArgumentTypeError:
        raise

parser = argparse.ArgumentParser()
parser.add_argument("--units", type=cnn_param, nargs="+")
args = parser.parse_args()

def build_model(out1, out2):
    X = 0
    model = Sequential()
    model.add(Conv2D(out1, kernel_size=(5,5), padding="same", input_shape=(28, 28, 1), activation="relu"))
    X += model.output_shape[1]**2*out1*1*25
    model.add(MaxPool2D())
    model.add(Conv2D(out2, kernel_size=(5,5), padding="same", activation="relu"))
    X += model.output_shape[1]**2*out2*out1*25
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, X

for out1, out2 in args.units:
    model, X = build_model(out1, out2)
    start = datetime.now()
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=2, batch_size=100)
    end = datetime.now()
    print(X, (end-start).total_seconds())