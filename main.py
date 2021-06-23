#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 14:33
# @Author  : Darren
# @Site    : 
# @File    : main.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchsummary import summary
import argparse
from model.STNNET import STN_Net
from util.dataloader import get_dataloader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


def train(net, epoch_nums, lr, train_dataloader, per_batch, device):
    # 使用训练模式
    net.train()
    # 选择梯度下降优化算法
    optimizer = optim.SGD(net.parameters(), lr=lr)
    # 训练模型
    for epoch in range(epoch_nums):
        for batch_idx, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            pred = net(data)
            loss = F.nll_loss(pred, label)
            loss.backward()
            optimizer.step()

            if batch_idx % per_batch == 0:
                print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(epoch, batch_idx * len(data),
                                                                             len(train_dataloader.dataset),
                                                                             100. * batch_idx / len(train_dataloader),
                                                                             loss.item()))


def evaluate(net, test_dataloader, device):
    with torch.no_grad():
        # 使用评估模式
        net.eval()
        eval_loss = 0
        eval_acc = 0
        for data, label in test_dataloader:
            data, label = data.to(device), label.to(device)
            pred = net(data)

            eval_loss += F.nll_loss(pred, label,
                                    size_average=False).item()
            pred_label = pred.max(1, keepdim=True)[1]
            eval_acc += pred_label.eq(label.view_as(pred_label)
                                      ).sum().item()

        eval_loss /= len(test_dataloader.dataset)
        print("evaluate set: Average loss: {:.4f},Accuracy:{}/{}({:.2f}%)\n".format(
            eval_loss, eval_acc, len(test_dataloader.dataset),
            100 * eval_acc / len(test_dataloader.dataset)))


def tensor_to_array(img_tensor):
    img_array = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = std * img_array + mean
    img = np.clip(img_array, 0, 1)
    return img


def visualize_stn(net, dataloader, device):
    with torch.no_grad():
        data = next(iter(dataloader))[0].to(device)

        input_tensor = data.cpu()
        t_input_tensor = net.stn(data).cpu()

        in_grid = tensor_to_array(torchvision.utils.make_grid(
            input_tensor))
        out_grid = tensor_to_array(torchvision.utils.make_grid(
            t_input_tensor))

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("input images")

        axarr[1].imshow(out_grid)
        axarr[1].set_title("stn transformed images")

        plt.show()


def parse_args():
    parse = argparse.ArgumentParser("config stn args")
    parse.add_argument("--lr", default=0.01,
                       type=float, help="learning rate")
    parse.add_argument("--epoch_nums", default=2,
                       type=int, help="iterated epochs")
    parse.add_argument("--use_stn", default=True,
                       type=bool, help="whether to use STN module")
    parse.add_argument("--batch_size", default=64,
                       type=int, help="batch size")
    parse.add_argument("--use_eval", default=True,
                       type=bool, help="whether to evaluate")
    parse.add_argument("--use_visual", default=True,
                       type=bool, help="visual STN transform image")
    parse.add_argument("--use_gpu", default=True,
                       type=bool, help="whether to use GPU")
    parse.add_argument("--show_net_construct", default=False,
                       type=bool, help="print net construct info")
    return parse.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # 加载数据集
    train_loader, test_loader = get_dataloader(args.batch_size)
    # 创建网络
    net = STN_Net(args.use_stn).to(device)
    # 打印网络的结构信息
    if args.show_net_construct:
        summary(net, (1, 28, 28))
    # 训练模型
    train(net, args.epoch_nums, args.lr, train_loader
          , args.batch_size, device)
    if args.use_eval:
        # 评估模型
        evaluate(net, test_loader, device)
    if args.use_visual:
        # 可视化展示效果
        visualize_stn(net, test_loader, device)
