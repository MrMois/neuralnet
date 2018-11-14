#!/usr/bin/python3

import cv2
import numpy as np


class Midpoint:

    def __init__(self, inner_margin, outer_margin, image_size):

        # position of midpoint and size of border
        self.inner_pos = [inner_margin, inner_margin]
        self.margin = inner_margin

        self.image_size = inner_image_size

    def next_position(self):

        # horizontal first scan
        if self.position[1] + self.inner_margin + 1 < self.image_size[1]:
            self.position[1] += 1

        # vertical second scan
        elif self.position[0] + self.inner_margin + 1 < self.image_size[0]:
            self.position[1] = 0
            self.position[0] += 1

        # reset, image scanned
        else:
            self.position = [self.inner_margin, self.inner_margin]

        return self.position


class Box:

    def __init__(self, source, margin, color=cv2.IMREAD_COLOR):

        if self is not OuterBox:
            self.img = cv2.imread(source, color)

        self.margin = margin
        self.length = 2*margin + 1
        self.shape = (self.length, self.length, 3)

        self.shape_1d = np.prod(self.shape)

    def next(self, midpoint_pos):

        y = [midpoint_pos[0]-self.margin, midpoint_pos[0]+self.margin+1]
        x = [midpoint_pos[1]-self.margin, midpoint_pos[1]+self.margin+1]

        return self.img[y[0]:y[1], x[0]:x[1]]


class OuterBox(Box):

    def __init__(self, source, margin, void_color):
        self.img = self.load_img(source, margin, void_color)
        Box.__init__(self, source, margin, color=cv2.IMREAD_GRAYSCALE)

    def load_img(self, source, margin, void_color):

        img_data = cv2.imread(source, 0)

        new_shape = (img_data.shape[0] + 2*margin,
                     img_data.shape[1] + 2*margin)
        # Add a border of void_color entries around the matrix
        new_data = np.ones(new_shape, dtype=int) * void_color
        new_data[margin:img_data.shape[0]+margin,
                 margin:img_data.shape[1]+margin] = img_data

        return new_data

    def next(self, midpoint_pos):

        y = [midpoint_pos[0], midpoint_pos[0]+2*self.margin+1]
        x = [midpoint_pos[1], midpoint_pos[1]+2*self.margin+1]

        return self.img[y[0]:y[1], x[0]:x[1]]


class Data:

    def __init__(self, source, inner_margin, outer_margin, void_color):

        self.inner_box = Box(source, inner_margin)
        self.outer_box = OuterBox(source, outer_margin, void_color)
        self.midpoint = Midpoint(inner_margin, self.inner_box.img.shape)

    def next(self):

        midpoint_pos = self.midpoint.next_position()
        inner_box = self.inner_box.next(midpoint_pos)
        outer_box = self.outer_box.next(midpoint_pos)

        return inner_box, outer_box


def main():

    d = Data("rotterdam.jpg", 50, 70, 255)

    for _ in range(10000):
        inner, outer = d.next()
        cv2.imshow("inner", inner)
        cv2.imshow("outer", outer)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
