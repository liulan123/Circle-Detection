from __future__ import division

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import operator
from progressbar import *
CANNY = 0
SOBEL = 1
CIRCLE = 0

def ReadImageGray(path):
    image = cv.imread(path, 0)
    return image

def ShowImage(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def GaussBlur(image, kernel_size=3):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

def EdgeDetect(image, method=CANNY, minv=120, maxv=255):
    if method == CANNY:
        return cv.Canny(image, minv, maxv)

def round(x):
    x_r = np.around(x)
    if x - x_r > 0.25:
        return x_r + 0.5
    if x_r - x > 0.25:
        return x_r - 0.5
    return x_r


def Hough(edge_points, height, width, shape=CIRCLE, scale_min=10, scale_max=20):
    if shape == CIRCLE:
        grid = {}
        x_range = np.arange(scale_min, height-scale_min, 3)
        y_range = np.arange(scale_min, width-scale_min, 3)
        total = x_range.size * y_range.size
        # r_range = np.arange(1, math.ceil(min(H, W) / 2))
        centers = itertools.product(x_range, y_range)
        pbar = ProgressBar().start()
        for i, center in enumerate(centers):
            pbar.update(int((i / (total - 1)) * 100))
            for edge_point in edge_points:
                distance = round(np.linalg.norm(edge_point - center))
                if distance >= scale_max or distance <= scale_min:
                    continue
                key = tuple(np.append(center, distance).tolist())
                if key in grid:
                    grid[key] += (1. / distance)
                else:
                    grid[key] = 1 / distance            
        pbar.finish()
        grid_sorted = sorted(grid.items(), key=lambda x:x[1], reverse=True)
        return grid_sorted[:30]