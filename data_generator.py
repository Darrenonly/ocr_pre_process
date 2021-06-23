#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/25 10:31
# @Author  : Darren
# @Site    : 
# @File    : data_generator.py
# @Software: PyCharm


import cv2
import numpy as np
import random
import os
from config.config import *


def generator(img_path):
    # 首先读入img
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (180, 32))
    # N对基准控制点
    N = 5
    points = []
    dx = int(180 / (N - 1))
    for i in range(2 * N):
        points.append((dx * i, 4))
        points.append((dx * i, 36))
    # 周围拓宽一圈
    img = cv2.copyMakeBorder(img, 4, 4, 0, 0, cv2.BORDER_REPLICATE)
    # 画上绿色的圆圈
    # for point in points:
    # 	cv2.circle(img, point, 1, (0, 255, 0), 2)
    tps = cv2.createThinPlateSplineShapeTransformer()

    sourceshape = np.array(points, np.int32)
    sourceshape = sourceshape.reshape(1, -1, 2)
    matches = []
    for i in range(1, N + 1):
        matches.append(cv2.DMatch(i, i, 0))

    # 开始随机变动
    newpoints = []
    PADDINGSIZ = 10
    for i in range(N):
        nx = points[i][0] + random.randint(0, PADDINGSIZ) - PADDINGSIZ / 2
        ny = points[i][1] + random.randint(0, PADDINGSIZ) - PADDINGSIZ / 2
        newpoints.append((nx, ny))
    print(points, newpoints)
    targetshape = np.array(newpoints, np.int32)
    targetshape = targetshape.reshape(1, -1, 2)
    tps.estimateTransformation(sourceshape, targetshape, matches)
    img = tps.warpImage(img)
    # path process
    (path, file_name) = os.path.split(img_path)
    (file, ext) = os.path.splitext(file_name)
    path = str(file + '_' + 'gen' + ext)
    # save img
    cv2.imwrite(fake_tps_image_path + path, img)


def perspective_transform(path):
    img = cv2.imread(path, 1)
    rows, cols, channels = img.shape
    scale_x = 0.3
    scale_y = 1 - scale_x
    p1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    # 上视图、下视图、左视图、右视图
    points = [np.float32([[int(cols / 3), int(rows / 2)], [int(cols * 2 / 3), int(rows / 2)], [cols, rows], [0, rows]]),
              np.float32(
                  [[0, 0], [cols, 0], [int(cols * 2 / 3), int(rows * 2 / 3)], [int(cols / 3), int(rows * 2 / 3)]]),
              np.float32([[int(cols / 4), int(rows / 3)], [cols, 0], [cols, rows], [int(cols / 4), int(rows * 2 / 3)]]),
              np.float32(
                  [[0, 0], [int(cols * 3 / 4), int(rows / 3)], [int(cols * 3 / 4), int(rows * 2 / 3)], [0, rows]])]
    mark = ['up', 'down', 'left', 'right']
    i = 0
    # path process
    (path, file_name) = os.path.split(path)
    (file, ext) = os.path.splitext(file_name)
    for pts in points:
        M = cv2.getPerspectiveTransform(p1, pts)
        dst = cv2.warpPerspective(img, M, (cols, rows))
        path = str(file + '_' + mark[i] + ext)
        i += 1
        # save img
        cv2.imwrite(fake_transform_image_path + path, dst)
        print(fake_transform_image_path + path)


# image_size = 288
#
#
# def order_points(pts):
#     center = pts.sum(axis=0) / 4
#     deltaxy = pts - np.tile(center, (4, 1))
#     rad = np.arctan2(deltaxy[:, 1], deltaxy[:, 0])
#     sortidx = np.argsort(rad)
#     return pts[sortidx]
#
#
# def four_point_transform(image, pts, w, h):
#     src = order_points(pts)
#     dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
#     M = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(image, M, (w, h))
#     return warped


def dataloader(file_path):
    lists = os.listdir(file_path)
    for list in lists:
        if "train" in file_path:
            generator(real_image_path + list)
        else:
            perspective_transform(file_path + list)


if __name__ == '__main__':
    # perspective_transform(real_image_path + 'img_21.jpg')
    dataloader(fake_tps_image_path)
    # dataloader(fake_tps_image_path)
    # perspective_transform(fake_image_path + 'img_21_gen.jpg')
