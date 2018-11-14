#!/usr/bin/python3

import numpy as np


# Activation functions


class Sigmoid:

    def __init__(self):
        self.compute = lambda x: 1. / (1. + np.exp(-x))
        self.gradient = lambda y: y * (1 - y)


# Cost functions


class Quadratic:

    def __init__(self):
        self.compute = lambda activation, target: 0.5*(activation - target)**2
        self.gradient = lambda activation, target: activation - target


# Testing only


def main():

    activation = Sigmoid()
    input = np.arange(4)
    a = activation.compute(input)
    print(a)


if __name__ == "__main__":
    main()
