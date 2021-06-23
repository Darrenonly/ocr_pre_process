#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 14:29
# @Author  : Darren
# @Site    : 
# @File    : STNNET.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F


class STN_Net(nn.Module):
    def __init__(self, use_stn=True):
        super(STN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # 用来判断是否使用STN
        self._use_stn = use_stn

        # localisation net
        # 从输入图像中提取特征
        # 输入图片的shape为(-1,1,28,28)
        self.localization = nn.Sequential(
            # 卷积输出shape为(-1,8,22,22)
            nn.Conv2d(1, 8, kernel_size=7),
            # 最大池化输出shape为(-1,1,11,11)
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # 卷积输出shape为(-1,10,7,7)
            nn.Conv2d(8, 10, kernel_size=5),
            # 最大池化层输出shape为(-1,10,3,3)
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # 利用全连接层回归\theta参数
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]
                                                    , dtype=torch.float))

    def stn(self, x):
        # 提取输入图像中的特征
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        # 回归theta参数
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x.size())
        # 根据输入图片计算变换后图片位置填充的像素值
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # 使用STN模块
        if self._use_stn:
            x = self.stn(x)
        # 利用STN矫正过的图片来进行图片的分类
        # 经过conv1卷积输出的shape为(-1,10,24,24)
        # 经过max pool的输出shape为(-1,10,12,12)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 经过conv2卷积输出的shape为(-1,20,8,8)
        # 经过max pool的输出shape为(-1,20,4,4)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
