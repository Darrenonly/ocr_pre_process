#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 14:31
# @Author  : Darren
# @Site    : 
# @File    : dataloader.py
# @Software: PyCharm


import torch
from torchvision import datasets, transforms


def get_dataloader(batch_size):
    # 加载数据集
    # 如果GPU可用就用GPU,否则用CPU
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    # 加载训练集
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=batch_size, shuffle=True)

    # 加载测试集
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(root="./data", train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader