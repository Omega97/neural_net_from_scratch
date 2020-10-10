from random import random
from math import exp, log


def random_weights(length, weight_range):
    return [(random() - random()) * weight_range for _ in range(length)]


def loop_gen(n):
    def f():
        i = 0
        while True:
            yield i % n
            i += 1
    return f


def gen_next_n(n):
    def f(itr):
        i = 0
        for a in itr:
            if i == n:
                break
            i += 1
            yield a
    return f


def i_print(itr, n=None, separator=None):
    print()
    for i in itr:
        if n is not None:
            n -= 1
            if n == 0:
                break
        print(i)
        if separator is not None:
            print(end=separator)


def d_print(dct: dict, length=16):
    print()
    for key in dct:
        print(f'{key:{length}} {dct[key]}')


def sign(x):
    if x > 0:
        return +1
    elif x < 0:
        return -1
    else:
        return 0


def smooth_max(v: list):
    """derivable max (overflow at x > 700)"""
    return log(sum([exp(i) for i in v]))


def soft_max(v: list):
    s = sum([exp(i) for i in v])
    return [exp(i)/s for i in v]
