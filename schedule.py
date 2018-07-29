# -*- coding: utf-8 -*-
import operator


def greedy_division(seq: list, k: int):
    sequence = sorted(seq, reverse=True)
    results = [list() for _ in range(k)]
    count_dict = [0 for _ in range(k)]
    for elem in sequence:
        ind, count = min(enumerate(count_dict), key=operator.itemgetter(1))
        results[ind].append(elem)
        count_dict[ind] += elem
    return results


"""
    def karmarkar2(seq: list):
        #Karmarkar-Karp algorithm when k = 2
        #:param seq: list of values to partition
        #:return: 2 partitions of seq
        def generator(queue: list):
            while len(queue) > 1:
                yield queue.pop(0), queue.pop(0)
        sequence = sorted(seq, reverse=True)
        stack = list()
        other = list()
        for v1, v2 in generator(sequence):
            diff = v1 - v2
            stack.append((v1, v2, diff))
            sequence.append(diff)
            sequence.sort(reverse=True)
        for v1, v2, diff in reversed(stack):
            if diff in sequence:
                sequence.remove(diff)
                sequence.append(v1)
                other.append(v2)
            else:
                other.remove(diff)
                other.append(v1)
                sequence.append(v2)
        return sequence, other
"""


def karmarkar(seq: list, k: int):
    def karmarkar_merge(seq1, seq2):
        return [i+j for i, j in zip(seq1, reversed(seq2))]

    def karmarkar_shrink(sequence):
        minimum = min(sequence)
        for i in range(len(sequence)):
            sequence[i] -= minimum

    def karmarkar_sort(ps, p):
        tmp = sorted(zip(ps, p), key=operator.itemgetter(0), reverse=True)
        return [i[0] for i in tmp], [i[1] for i in tmp]

    partitions_states = [[i] + [0 for _ in range(k-1)] for i in seq]
    partitions_states.sort(key=operator.itemgetter(0), reverse=True)
    partitions = [[[elem]] + [list() for _ in range(k-1)] for elem, *others in partitions_states]

    while len(partitions_states) > 1:
        ps1, ps2 = partitions_states.pop(0), partitions_states.pop(0)
        p1, p2 = partitions.pop(0), partitions.pop(0)
        ps = karmarkar_merge(ps1, ps2)
        p = karmarkar_merge(p1, p2)
        karmarkar_shrink(ps)
        ps, p = karmarkar_sort(ps, p)
        for ind, (_p, _ps) in enumerate(zip(partitions, partitions_states)):
            if _ps[0] <= ps[0]:
                partitions.insert(ind, p)
                partitions_states.insert(ind, ps)
                break
        else:
            partitions.append(p)
            partitions_states.append(ps)
    return partitions[0]


def initial_plan(workloads, deadline):
    terminate = False
    if deadline < max(workloads):
        raise ValueError("deadline small than max of workloads")
    k = int(sum(workloads) / deadline) + 1
    while not terminate:
        plan = karmarkar(workloads, k)
        temp_max = max([sum(i) for i in plan])
        if temp_max > deadline:
            k = k + 1
        else:
            terminate = True
    return plan


def _plan_fit(division, deadline):
    return max([sum(loads) for loads in division]) <= deadline


def optimization_plan(workloads, deadline):
    if deadline < max(workloads):
        raise ValueError("deadline small than max of workloads")
    if deadline >= sum(workloads):
        return 1
    if len(workloads) <= 2:
        return len(workloads)
    invalid, valid = 1, len(workloads)
    while invalid < valid - 1:
        k = (invalid + valid) // 2
        division = karmarkar(workloads, k)
        if _plan_fit(division, deadline):
            valid = k
        else:
            invalid = k
    division = karmarkar(workloads, valid)
    return division
