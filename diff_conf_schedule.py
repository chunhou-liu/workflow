# -*- coding: utf-8 -*-
import copy
import operator
class Case(object):
    def __init__(self, color, capacity):
        self.color = color
        self.capacity = capacity
        self.objects = []

    @property
    def left_volume(self):
        return self.capacity - sum([obj[1] for obj in self.objects])
    
    def put(self, obj):
        self.objects.append(obj)


def arg_min(lst, key):
    return min(enumerate(lst), key=lambda x: key(x[1]))[0]


def arg_max(lst, key):
    return max(enumerate(lst), key=lambda x: key(x[1]))[0]


def mapping_sort(lst, key, reversed=False):
    sorted_list = sorted(enumerate(lst), key=lambda x: key(x[1]), reverse=reversed)
    return dict([(new_index, old_index) for new_index, (old_index, val) in enumerate(sorted_list)]), [val for ind, val in sorted_list]


def deadline_constraint(costs, prices, deadline):
    """
        params:
            costs: NxM 2-d array, N tasks and M kinds machines
            prices: M kinds machines price
        return:
            deployment of each type of machines
    """
    def calculate_average_cost(task):
        "task: 1xM 1-d array, cost of task"
        return sum(task) / len(task)

    def calculate_price_performance_ratio(tasks, prices):
        return [[1/(cost*price) for cost, price in zip(task, prices)] for task in tasks]
        #return [[cost/price for cost, price in zip(task, prices)] for task in tasks]

    def coloring_object(cp_ratio, prices):
        max_ratio, min_price, color = min(cp_ratio), max(prices), -1
        for ind, (ratio, price) in enumerate(zip(cp_ratio, prices)):
            if ratio > max_ratio or (ratio==max_ratio and price <= min_price):
                max_ratio, min_price, color = ratio, price, ind
        return color
    # sort tasks in order of average cost
    index_mapping, tasks = mapping_sort(costs, calculate_average_cost)
    # calculate price-performance ratio of each task and machine
    cp_ratios = calculate_price_performance_ratio(tasks, prices)
    packages = dict([(i, list()) for i in range(len(prices))])
    for ind, (task, cp_ratio) in enumerate(zip(tasks, cp_ratios)):
        # coloring object
        color = coloring_object(cp_ratio, prices)
        #if len(packages[color]) == 0:
        #    packages[color].append(Case(color, deadline))
        for case in packages[color]:
            if case.left_volume >= task[color]:
                case.put((ind, task[color]))
                break
        else:
            # when there is no case available
            case = Case(color, deadline)
            case.put((ind, task[color]))
            packages[color].append(case)
        packages[color].sort(key=operator.attrgetter("left_volume"))

    return dict([(color, [[index_mapping[ind] for ind, cost in case.objects] for case in cases]) for color, cases in packages.items()])
        


if __name__ == "__main__":
    params = [
        (1,2,3,4),
        (2,3),
        (4,5,6),
        (7,9,10)
    ]
    costs = [
        [1,2,4,5,6],
        [4,5,7,3,5],
        [1,2,1,7.5,1],
        [1,2,3,5,2]
    ]
    prices = [
        4,5,6,10,6
    ]
    deployment = deadline_constraint(costs, prices, 100)
    for machine_type in deployment:
        print("machine type",machine_type, "price", prices[machine_type])
        for ind, machine in enumerate(deployment[machine_type], start=1):
            print("\tmachine",ind)
            for task_index in machine:
                print("\t\t", params[task_index], costs[task_index][machine_type])