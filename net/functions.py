#!/usr/bin/python3

import numpy as np


# Activation functions


class Sigmoid:

    def compute(x):
        return 1. / (1. + np.exp(-x))

    def gradient(y):
        return y * (1 - y)


# Cost functions


class Quadratic:

    def compute(activation, target):
        return 0.5*(activation - target)**2

    def gradient(activation, target):
        return activation - target


# Testing only


def main():

    activation = Sigmoid
    input = np.arange(4)
    a = activation.compute(input)
    print(a)


if __name__ == "__main__":
    main()
