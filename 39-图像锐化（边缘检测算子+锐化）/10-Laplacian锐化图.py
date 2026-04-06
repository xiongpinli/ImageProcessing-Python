# Copyright 2026. All rights reserved.
# Author: xiongpingli
# Date: 2026/4/6
# Time: 下午8:02
# descript:
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('lena.png')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 灰度化
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==================== 拉普拉斯 边缘检测 ====================
dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
Laplacian_edge = cv2.convertScaleAbs(dst)  # 边缘图

# ==================== 拉普拉斯 锐化（核心！） ====================
# # 锐化公式：锐化图 = 原图 + 边缘图
# sharpened = cv2.addWeighted(grayImage, 1, Laplacian_edge, 1, 0)

# 或者用 numpy 加法（效果一样）
sharpened = cv2.add(grayImage, Laplacian_edge)

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']

# ==================== 显示结果 ====================
titles = ['原始灰度图', '拉普拉斯边缘', '拉普拉斯锐化图']
images = [grayImage, Laplacian_edge, sharpened]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()