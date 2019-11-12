#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from skimage import feature
from skimage.feature import local_binary_pattern

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
    tmp[tmp >= val] = 1
    tmp[1, 1] = 0
    pow2_circ = make_pow2_circle()
    res = pow2_circ * tmp
    return np.add.reduce(res, (0, 1))


def apply_lbp_on_image(img):
    tmp = np.zeros(img.shape)

    for i in progressbar.progressbar(range(1, img.shape[0] - 1)):
        for j in range(1, img.shape[1] - 1):
            tmp[i, j] = compute_elm_of_area_3x3(img[i - 1:i + 2, j - 1:j + 2])
    return tmp


def openImg(path):
    iimm = np.array(cv2.imread(path))
    iimm = cv2.cvtColor(iimm, cv2.COLOR_BGR2RGB)
    iimm = cv2.cvtColor(iimm, cv2.COLOR_RGB2GRAY)
    return iimm.astype(np.float32)


def plot_img(im):
    fig, ax = plt.subplots(figsize=(19, 11))
    im = im.astype(np.uint8)
    ax.imshow(im, aspect='auto', interpolation='nearest')
    plt.show()


def comparison_func(img):
    return feature.local_binary_pattern(img, 8, 1, method="default")


def main():
    img = openImg("./Dataset/12084.jpg")
    res = apply_lbp_on_image(img)
    plot_img(res)


main()
