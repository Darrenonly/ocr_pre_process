#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/28 16:14
# @Author  : Darren
# @Site    : 
# @File    : camera_test.py
# @Software: PyCharm


# -*- coding:utf8 -*-
import cv2


def make_photo():
    """使用opencv拍照"""
    cap = cv2.VideoCapture(0)  # 默认的摄像头
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("capture", frame)  # 弹窗口
            # 等待按键q操作关闭摄像头
            if cv2.waitKey(1) & 0xFF == ord('q'):
                file_name = "my.jpg"
                cv2.imwrite(file_name, frame)
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def make_video():
    """使用opencv录像"""
    cap = cv2.VideoCapture(0)  # 默认的摄像头
    # 指定视频代码
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter('myvideo.avi', fourcc, 20.0, (640, 480))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            #
            cv2.imshow('frame', frame)
            # 等待按键q操作关闭摄像头
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    make_video()