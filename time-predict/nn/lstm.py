# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import LSTM, Dense


def build_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(28, 28)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model