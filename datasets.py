from data import *
from random import random as r


def binary_dataset(operator):
    """
    general binary operator dataset
    :param operator: lambda expression
    :return: DataSet
    """
    def dataset():
        v = []
        for x1 in range(2):
            for x2 in range(2):
                y = operator(x1, x2)
                v += [{'x': [x1, x2], 'y': [y]}]
        return DataSet(v)
    return dataset()


def xor_dataset():
    return binary_dataset(lambda a, b: a ^ b)


def and_dataset():
    return binary_dataset(lambda a, b: a & b)


def noisy_fit(size=100, length=2, noise=.1):
    """some random noisy data to fit"""
    def dataset():
        v = []
        for _ in range(size):
            x = [round(r()-r(), 2) for __ in range(length)]
            y = round(sum([i**2 for i in x]) + (r() - r()) * noise, 2)
            v += [{'x': x, 'y': [y]}]
        return DataSet(v)
    return dataset()


def add_noise(noise=.1):    # todo check
    def _add_noise(data_gen):
        def f():
            for data in data_gen:
                yield {'x': data['x'], 'y': data['y'] + (random() - random()) * noise}
        return f()
    return _add_noise


def noisy_xor(noise_prob=.1):
    """usually xor, sometimes random """
    def f():
        while True:
            x1 = randrange(2)
            x2 = randrange(2)
            y = x1 ^ x2 if random() >= noise_prob else randrange(2)
            yield {'x': [x1, x2], 'y': [y]}
    return DataGenerator(f())


def mod_dataset():

    def f():
        while True:
            x = [randrange(2) for _ in range(4)]
            y = [x[i * 2] & x[i * 2 + 1] for i in range(len(x) // 2)]
            y = [y[i * 2] | y[i * 2 + 1] for i in range(len(y) // 2)]
            yield {'x': x, 'y': y}

    return DataGenerator(f())
