#!/usr/bin/python3

"""
# TODO:
    momentum
    debug other act.funcs gradient in backward
"""

import numpy as np


class Layer:

    def __init__(self, f_activation, weights=None, size=None):

        self.f_activation = f_activation

        if weights is not None:
            self.weights = weights

        # create new weights
        elif size is not None:
            self.weights = Layer.create_weights(size)

    # size should be [output_size, input_size]
    @staticmethod
    def create_weights(size):

        weights = 2 * np.random.rand(size[0], size[1]) - 1
        # extra row for bias
        biases = 0.5 * np.random.rand(size[0], 1) + 0.5
        # append bias vector to weight matrix
        return np.c_[weights, biases]

    # single input forward
    def forward(self, input):

        self.input = input
        input_biased = np.r_[input, [1]]  # add bias

        self.output = np.dot(self.weights, input_biased)
        self.activation = self.f_activation.compute(self.output)

        return self.output, self.activation

    # single target backward
    def backward(self, w_next, delta_next):

        delta = np.dot(w_next[:, :-1].T, delta_next)
        # functions like sigmoid use activation instead of output
        self.delta = self.f_activation.gradient(self.activation)
        self.delta *= delta

        # next layer needs weights as w_next
        return self.delta, self.weights

    # correct weights
    def apply_delta(self, learning_rate):

        self.delta_w = np.c_[
                np.dot(
                    self.delta.reshape((len(self.delta), 1)),
                    self.input.reshape(1, (len(self.input)))),
                self.delta]

        self.weights += -learning_rate * self.delta_w


class OutputLayer(Layer):

    def __init__(self, f_activation, f_cost, weights=None, size=None):

        Layer.__init__(self, f_activation, weights, size)
        self.f_cost = f_cost

    # single target backward
    def backward(self, target):

        cost = self.f_cost.compute(self.activation, target)
        error = self.f_cost.gradient(self.activation, target)
        # functions like sigmoid use act instead of out
        self.delta = self.f_activation.gradient(self.activation)
        self.delta *= error

        return self.delta, self.weights, cost
