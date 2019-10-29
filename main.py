#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


def grad(image, i, j):
    i = min(i, image.shape[0] - 2)
    j = min(j, image.shape[1] - 2)
    return np.sqrt((image[i + 1, j] - image[i - 1, j]) ** 2 + (image[i, j + 1] - image[i, j - 1]) ** 2)


def compute_gradient(image):
    res = [grad(image, i, j) for i in range(image.shape[0]) for j in range(image.shape[1])]
    return np.array(res).reshape(image.shape)


def openImg():
    iimm = np.array(cv2.imread("./Dataset/108005.jpg"))
    iimm = cv2.cvtColor(iimm, cv2.COLOR_BGR2RGB)
    iimm = cv2.cvtColor(iimm, cv2.COLOR_RGB2LAB)
    return iimm.astype(np.float32)


def plot_img(im):
    im = im.astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_LAB2RGB)
    plt.imshow(im)
    plt.show()
    print("Done.")


def make_color_points(im, k):
    im_grad = compute_gradient(im)
    img_height = im.shape[0]
    img_width = im.shape[1]
    length = img_width * img_height
    S = int(np.sqrt(length / k))
    close_to_origin = 2
    positions = [((i / k * length) + (length / (2 * k * close_to_origin) * (random.random() * 2 - 1))) for i in
                 range(k)]
    positions = np.array([[int(elm // img_width), int(elm % img_width)] for elm in positions])
    single_grad_values = np.add.reduce(im_grad, 2)
    for i, elm in enumerate(positions):
        xx = elm[0]
        yy = elm[1]
        minus = S // 2
        maxus = minus + 1
        minxx = max(0, int(xx - minus))
        maxxx = min(img_width - maxus, int(xx + maxus))
        minyy = max(0, int(yy - minus))
        maxyy = min(img_width - maxus, int(yy + maxus))
        window = single_grad_values[minxx:maxxx, minyy:maxyy]
        relative_pos = 0
        if window.any():
            relative_pos = window.argmin()
        minposx = (relative_pos // S) - S // 2
        minposy = (relative_pos % S) - S // 2
        positions[i][0] += minposx
        positions[i][1] += minposy
    positions[positions < 0] = 0
    positions[..., 0][positions[..., 0] > img_height] = img_height - 1
    positions[..., 1][positions[..., 1] > img_width] = img_width - 1
    return positions


def plot_points_to_img(im, color_points):
    plt.imshow(compute_gradient(im).astype(np.uint8))
    for elm in color_points:
        plt.scatter(elm[0], elm[1], marker='o')
    plt.show()


def euclid_dist(arr1, arr2):
    return np.sqrt(np.add.reduce(arr1 - arr2))


def distance_color(coord1, color1, coord2, color2, s):
    m = 10
    dlab = euclid_dist(color1, color2)
    dxy = euclid_dist(coord1, coord2)
    cooef = m / s
    return dlab + cooef * dxy


def make_image(im, color_points, k):
    img_height = im.shape[0]
    img_width = im.shape[1]
    length = img_width * img_height
    S = int(np.sqrt(length / k))
    res = np.array(im)
    set_color_point = set([(elm[0], elm[1]) for elm in color_points])

    dico = {}
    for elm in color_points:
        if not dico.get(elm[0]):
            dico[elm[0]] = {}
        elm[0] = min(elm[0], img_height - 1)
        elm[1] = min(elm[1], img_width - 1)
        dico[elm[0]][elm[1]] = im[elm[0], elm[1]]

    for i in range(img_height):
        for j in range(img_width):
            minx = max(0, i - S)
            maxx = min(img_height - 1, i + S)
            miny = max(0, j - S)
            maxy = min(img_width - 1, j + S)
            clusterList = [[a, b] for a in range(minx, maxx) for b in range(miny, maxy) if (a, b) in set_color_point]
            curr_elm_coord = np.array([i, j])
            curr_elm_color = np.array(im[i, j])
            if clusterList.__len__() == 0:
                continue
            closest_coord = np.array(clusterList[0])
            closest_color = np.array(im[closest_coord[0], closest_coord[1]])
            last_distance = distance_color(curr_elm_coord, curr_elm_color, closest_coord, closest_color, S)
            for elm in clusterList:
                center_coord = np.array(elm)
                center_color = np.array(im[closest_coord[0], closest_coord[1]])
                tmp_dist = distance_color(curr_elm_coord, curr_elm_color, center_coord, center_color, S)
                if tmp_dist < last_distance:
                    last_distance = tmp_dist
                    closest_color = center_color
                    closest_coord = center_coord

            res[i, j] = closest_color
    return res


def main():
    im = openImg()

    color_points = make_color_points(im, 3000)
    im = make_image(im, color_points, 3000)
    # print(color_points)
    # img_points = add_points_to_img(im, color_points)

    plot_img(im)


if __name__ == '__main__':
    main()
