from math import tanh as tanh_
from math import exp, log


class ActFun:

    def __init__(self, fun, der=None, q_der=None):
        self.fun = fun
        self.der = der
        self.q_der = q_der

    def __call__(self, x):
        return self.fun(x)

    def der(self, x):
        if self.der is None:
            raise NotImplementedError('derivative not implemented')
        return self.der(x)

    def quick_d(self, x):
        if self.q_der is None:
            raise NotImplementedError('quick derivative not implemented')
        return self.q_der(x)


def tanh():
    return ActFun(fun=tanh_,
                  q_der=lambda x: 1-x**2)


def sigmoid():
    return ActFun(fun=lambda x: (tanh_(x/2) + 1) / 2,
                  q_der=lambda x: (1-x) * x)


def prelu(eta=.1):
    return ActFun(fun=lambda x: x if x >= 0 else x * eta,
                  q_der=lambda x: 1 if x >= 0 else eta)


def softplus():
    return ActFun(fun=lambda x: log(exp(x)+1),
                  der=lambda x: 1/(1+exp(-x)))
