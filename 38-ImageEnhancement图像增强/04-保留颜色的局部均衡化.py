# Copyright 2026. All rights reserved.
# Author: xiongpingli
# Date: 2026/3/29
# Time: 下午4:09
# descript:

import cv2
import numpy as np

# 1. 读取原始彩色图
img_bgr = cv2.imread('test01.png')
# 2. 转 YCbCr（Y=亮度，Cb/Cr=色度，完全保留）
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img_ycrcb)  # 分离通道

# 3. 只对亮度Y做均衡化/CLAHE
# 全局均衡化
# y_eq = cv2.equalizeHist(y)
# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
y_eq = clahe.apply(y)
# 4. 合并处理后的Y + 原始Cb、Cr（彩色完全保留）
result = cv2.merge((y_eq, cr, cb))
# 5. 转回BGR彩色图
img_color_enhanced = cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)
cv2.imshow('img_bgr', img_bgr)
cv2.imshow('img_color_enhanced', img_color_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
