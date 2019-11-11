#!/usr/bin/env python3
import numpy as np


def make_pow2_circle():
    tmp = np.full((3, 3), 0)
    tmp[1, 2] = 2 ** 0
    tmp[2, 2] = 2 ** 1
    tmp[2, 1] = 2 ** 2
    tmp[2, 0] = 2 ** 3
    tmp[1, 0] = 2 ** 4
    tmp[0, 0] = 2 ** 5
    tmp[0, 1] = 2 ** 6
    tmp[0, 2] = 2 ** 7
    return tmp

### This has to be applied on the whole image
def compute_elm_of_area_3x3(roi):
    tmp = np.array(roi)
    val = tmp[1, 1]
    tmp[tmp < val] = 0
    tmp[tmp > val] = 1
    tmp[1, 1] = 0
    pow2_circ = make_pow2_circle()
    res = pow2_circ * tmp
    return np.add.reduce(res, (0, 1))



# tt = np.arange(9).reshape((3, 3))
# ru = compute_elm_of_area_3x3(tt)
# print(ru)
