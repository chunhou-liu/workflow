# -*- coding: utf-8 -*-
import os
from tensorflow.examples.tutorials.mnist import input_data


_BASE_DIR = os.path.dirname(__file__)
mnist = input_data.read_data_sets(os.path.join(_BASE_DIR, "mnist/"), one_hot=False)
