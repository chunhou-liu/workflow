# -*- coding: utf-8 -*-
import os
import pickle
from collections import namedtuple
import numpy as np


_BASE_PATH = os.path.dirname(__file__)
_cifar10_train = os.path.join(_BASE_PATH, "cifar10/train")
_cifar10_test = os.path.join(_BASE_PATH, "cifar10/test")


if not os.path.exists(_cifar10_train):
    import requests
    sess = requests.get("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", stream=True)
    with open("cifar-10-python.tar.gz", "wb") as f:
        for ind, chunk in enumerate(sess.iter_content(chunk_size=1024*1024), start=1):
            f.write(chunk)
            print(ind, "KB")

def cifar10():
    with open(_cifar10_train, "rb") as f:
        cifar10_train_dict = pickle.load(f, encoding="bytes")
    with open(_cifar10_test, "rb") as f:
        cifar10_test_dict = pickle.load(f, encoding="bytes")
    cifar = namedtuple("cifar10", ["train", "test"])
    dataset = namedtuple("dataset", ["images", "labels"])
    train_dataset = dataset(images=cifar10_train_dict[b"data"], labels=np.asarray(cifar10_train_dict[b"fine_labels"]))
    test_dataset = dataset(images=cifar10_test_dict[b"data"], labels=np.asarray(cifar10_test_dict[b"fine_labels"]))
    return cifar(train=train_dataset, test=test_dataset)


"""
if __name__ == "__main__":
    cifar = cifar10()
    print(cifar.train.images, len(cifar.train.images))
    print(cifar.train.labels, len(cifar.train.labels))
    print(cifar.test.images, len(cifar.test.images))
    print(cifar.test.labels, len(cifar.test.labels))
"""