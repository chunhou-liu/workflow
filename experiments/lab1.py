# -*- coding: utf-8 -*-
import argparse
import itertools
from models.cnn import CNNNetwork
import schedule


def tree_complexity(max_depth):
    if max_depth < 40:
        return 0.11924525 * max_depth + 8.522656142857143
    return 14.8


def svm_complexity(n):
    return 3.95412256e-07* n**2 + 5.584966244708394


def cnn_complexity(c1, c2):
    cnn = CNNNetwork(c1, c2)
    return 4.20293365e-05*cnn.X + 144.7993470956009

def cnn_params(s):
    try:
        return tuple(map(int, s.split(",")))
    except:
        raise argparse.ArgumentTypeError("must be a tuple of integers")


parser = argparse.ArgumentParser()
parser.add_argument("--tree", nargs="+", type=int)
parser.add_argument("--cnn", nargs="+", type=cnn_params)
args = parser.parse_args()

tree_time = dict([(i, tree_complexity(i)) for i in args.tree])
cnn_time = dict([(cnn_complexity(*i), i) for i in args.cnn])
svm_time = dict([(svm_complexity(55000), 55000)])

print("%"*50)
for key, val in tree_time.items():
    print(key, val)
print("%"*50)
for key, val in cnn_time.items():
    print(key, val)
print("%"*50)
for key, val in svm_time.items():
    print(key, val)
print("%"*50)

times = [i for i in itertools.chain(tree_time.values(), cnn_time.keys(), svm_time.keys())]
plan = schedule.initial_plan(times, max(times))
for machine in plan:
    print(machine)

# python -m experiments.lab1 --tree 30 50 70 90 100 500 --cnn 32,64 32,120 60,80 6,16
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
30 12.100013642857142
50 14.8
70 14.8
90 14.8
100 14.8
500 14.8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
301.88819519800086 (32, 64)
422.3947088108009 (32, 120)
503.89799815160086 (60, 80)
154.8863878556009 (6, 16)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
1201.7070406447083 55000
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[1201.7070406447083]
[422.3947088108009, 301.88819519800086, 12.100013642857142]
[503.89799815160086, 154.8863878556009, 14.8, 14.8, 14.8, 14.8, 14.8]
"""