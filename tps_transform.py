#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 15:24
# @Author  : Darren
# @Site    : 
# @File    : tps_transform.py
# @Software: PyCharm


import cv2
import numpy as np
import random
import os

name = 1281
out_path = './data/cycle_image/output_image/'
img_path = './data/cycle_image/0428/images/' + str(name) + '.jpg'
lables = './data/cycle_image/0428/labels/' + str(name) + '.txt'


def get_points(image_path, lables_path, output_path, N, threshold):
    """
    输入图像路径和标签文件路径
    :param threshold:
    :param output_path:
    :param image_path:
    :param lables_path:
    :param N: points number
    :return points:
    """
    # 首先读入img
    img = cv2.imread(image_path)
    label = open(lables_path, 'r', encoding='utf8')

    lines = label.readlines()
    (file_path, file_name) = os.path.split(image_path)
    (file_first_name, ext) = os.path.splitext(file_name)
    index = 0
    try:
        for line in lines:
            points = []
            point_ = line.split(',')[0:-1]
            for i in range(len(point_) // 2):
                points.append((int(point_[2 * i]), int(point_[2 * i + 1])))

            roi_img, obj_points = points_process(img, points, N, threshold)
            roi_img = tps_trans(roi_img, obj_points, N)
            filename = output_path + file_first_name + '_' + str(index) + ext
            cv2.imwrite(filename, roi_img)
            index += 1
    except Exception as e:
        print('BugReason:', e)


def tps_trans(img, points, N):
    """
    输入裁切后的含目标区域的最小矩形及坐标关键点，经TPS变换矫正为水平图像
    :param N: points numbers
    :param img: Objection Area
    :param points: Objection points
    :return None:
    """
    h, w, c = img.shape
    # dim = (4*h, h)
    tps = cv2.createThinPlateSplineShapeTransformer()

    sourceshape = np.array(points, np.int32)
    sourceshape = sourceshape.reshape(1, -1, 2)

    matches = []
    # N = len(sourceshape)
    for i in range(0, N):
        matches.append(cv2.DMatch(i, i, 0))

    # 开始变动
    newpoints = []
    N = N // 2
    dx = int(w / (N - 1))
    for i in range(0, N):
        newpoints.append((dx * i, 2))
    for i in range(N - 1, -1, -1):
        newpoints.append((dx * i, h - 2))
    print('-------------------源关键点----------------------')
    print(points)
    print('-------------------TPS变换关键点----------------------')
    print(newpoints)
    targetshape = np.array(newpoints, np.int32)
    targetshape = targetshape.reshape(1, -1, 2)
    tps.estimateTransformation(targetshape, sourceshape, matches)
    roi_img_ = tps.warpImage(img)

    for point in newpoints:
        cv2.circle(roi_img_, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)

    cv2.namedWindow('img2', cv2.WINDOW_FREERATIO)
    # 拉伸到4：1比例
    # roi_img_ = cv2.resize(roi_img_, dim, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('img2', roi_img_)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return roi_img_


def circle_fitness_demo(x, y, num):
    # 多项式曲线生成
    poly = np.poly1d(np.polyfit(x, y, 3))
    print(poly)

    # 绘制拟合曲线
    # for t in range(30, 250, 1):
    #     y_ = np.int(poly(t))
    #     cv2.circle(image, (t, y_), 1, (0, 0, 255), 1, 8, 0)
    return poly


def del_repeat(repeat_list):
    new_list = list(dict.fromkeys(repeat_list))
    return new_list


def points_process(img, points, N, threshold):
    """
    :param img: 原始图像
    :param points: 目标文字坐标
    :param N: 定位关键点个数
    :param threshold: 定位阈值
    :return: 目标区域图像， 定位关键点
    """
    final_points = []
    # 对坐标排序，先按x轴排序，再按y轴排序
    sort_points = sorted(points, key=lambda points: (points[0], points[1]))
    sort_points = del_repeat(sort_points)

    # 计算坐标集的最小外接矩形
    x, y, w, h = cv2.boundingRect(np.array(points))
    roi_img = img[y:y + h, x:x + w]

    for i in range(len(sort_points)):
        sort_points[i] = (sort_points[i][0] - x, sort_points[i][1] - y)

    points_num = int(N // 2)
    dx = int(w // (points_num - 1))
    tem_point = []
    up_points = []
    down_points = []
    points_index = []
    for sort_point in sort_points:
        points_index.append(sort_point[0])
    points_index = del_repeat(points_index)

    try:
        for n in range(points_num):
            _x = n * dx
            for x_i in range(_x - 5, _x + 5):
                if x_i in points_index:
                    for sort_point in sort_points:
                        if sort_point[0] == x_i:
                            tem_point.append(sort_point)
                #
                # else:
                #     x_i += 1
            sort_index = sorted(tem_point, key=lambda k: k[1])
            up_points.append(sort_index[0])
            down_points.append(sort_index[-1])
            tem_point.clear()

        final_points.append(up_points)
        down_points = sorted(down_points, reverse=True)
        final_points.append(down_points)
        return roi_img, final_points
    except Exception as e:
        print('Reason:', e)


if __name__ == '__main__':
    get_points(img_path, lables, out_path, 10, 3)
