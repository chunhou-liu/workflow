# -*- coding: utf-8 -*-
import os
from tensorflow.examples.tutorials.mnist import input_data


_BASE_DIR = os.path.dirname(__file__)
def mnist(one_hot=False):
    return input_data.read_data_sets(os.path.join(_BASE_DIR, "mnist/"), one_hot=one_hot)
