#!/usr/bin/python3

import numpy as np

data = np.arange(81).reshape((9, 9))

data_shape = data.shape

margin = 2
length = 5
void_color = 81

new_shape = (data_shape[0] + 2*margin, data_shape[1] + 2*margin)
new_data = np.ones(new_shape, dtype=int) * void_color

print(data)

print(data_shape)

print(new_data)

print(new_data.shape)

new_data[margin:data_shape[0]+margin, margin:data_shape[1]+margin] = data

print(new_data)

midpoint_pos = [1, 1]

y = [midpoint_pos[0]-margin, midpoint_pos[0]+margin+1]
x = [midpoint_pos[1]-margin, midpoint_pos[1]+margin+1]
