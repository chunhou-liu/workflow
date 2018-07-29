# -*- coding: utf-8 -*-
from datetime import datetime
from datasets.mnist import mnist


class Evaluator(object):
    def __init__(self, model, *params, logfile="log.txt"):
        self.Model = model
        self.params = params
        self.logfile = logfile

    def evaluate(self):
        for param in self.params:
            model = self.Model(*param)
            start = datetime.now()
            model.train(mnist, train_steps=10000)
            end = datetime.now()
            acc = model.test(mnist.test.images, mnist.test.labels)
            with open(self.logfile, "a") as f:
                print(param, model.var, (end-start).total_seconds(), acc, sep='\t', file=f)
                print(param, model.var, (end - start).total_seconds(), acc, sep='\t')


if __name__ == "__main__":
    from models.lenet import LeNet
    params = [(i, j, k) for i in [6, 16, 32, 64] for j in [16, 32, 64] for k in [16, 32, 64, 120]]
    for param in params:
        evaluator = Evaluator(LeNet, param)
        evaluator.evaluate()
