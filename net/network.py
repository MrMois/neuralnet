#!/usr/bin/python3

"""
# TODO:
    load/save
"""


import numpy as np
from layer import Layer, OutputLayer


class Network:

    def __init__(self, f_activation, f_cost, layers=None, struct=None):

        self.f_activation = f_activation
        self.f_cost = f_cost

        if layers is not None:
            self.layers = layers

        elif struct is not None:
            self.layers = Network.create_layers(f_activation,
                                                f_cost, struct)

    @staticmethod
    def create_layers(f_activation, f_cost, struct):

        layers = []

        for input, output in zip(struct, struct[1:-1]):
            layer = Layer(f_activation, size=[output, input])
            layers.append(layer)

        layer = OutputLayer(f_activation, f_cost,
                            size=[struct[-1], struct[-2]])
        layers.append(layer)

        return layers

    # deep forward
    def forward(self, input):

        activation = np.copy(input)

        for l in self.layers:
            _, activation = l.forward(activation)

        return activation  # network output

    # deep backward
    def backward(self, target):

        delta, weights, current_cost = self.layers[-1].backward(target)

        for l in reversed(self.layers[:-1]):
            delta, weights = l.backward(delta_next=delta, w_next=weights)

        return current_cost

    def apply_delta(self, learning_rate):

        for l in self.layers:
            l.apply_delta(learning_rate)

    def train(self, input, target, learning_rate):

        activation = self.forward(input)
        current_cost = self.backward(target)

        self.apply_delta(learning_rate)

        return activation, current_cost


# Testing only

def Mininet_test():

    from functions import Sigmoid, Quadratic

    w_h = np.ones((2, 2)) * 0.5
    w_o = np.ones((1, 3)) * 0.5

    f_activation = Sigmoid()
    f_cost = Quadratic()

    layers = [Layer(f_activation, weights=w_h),
              OutputLayer(f_activation, f_cost, weights=w_o)]

    net = Network(f_activation, f_cost, layers=layers)

    input = [1]

    act, cost = net.train(input=input, target=input, learning_rate=1)

    print(act, cost)

    for l in net.layers:
        print(l.delta_w)


def XOR_test():

    import random
    from functions import Sigmoid, Quadratic

    f_activation = Sigmoid()
    f_cost = Quadratic()
    struct = [2, 4, 3, 1]

    net = Network(f_activation, f_cost, struct=struct)

    learning_rate = 0.1

    while True:

        input = [random.getrandbits(1), random.getrandbits(1)]

        if input[0] != input[1]:
            target = [1]
        else:
            target = [0]

        act, cost = net.train(input, target, learning_rate)

        print(cost)


if __name__ == "__main__":
    XOR_test()
