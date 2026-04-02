
import cv2  
import numpy as np  
import matplotlib.pyplot as plt

# 相比全局均衡化（cv2.equalizeHist()），CLAHE 会将图像划分为多个小区域（tiles），
# 在每个区域内独立做均衡化，同时限制对比度上限，避免噪声被过度放大。
# 最后通过插值消除块边界，得到更自然、细节更丰富的增强效果。
#读取图片
img = cv2.imread('test01.png')

#灰度转换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#局部直方图均衡化处理
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))

#将灰度图像和局部直方图相关联, 把直方图均衡化应用到灰度图 
result = clahe.apply(gray)
result1 = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
cv2.imshow('origin', gray)
cv2.imshow('result', result)
cv2.imshow('result1', result1)
cv2.waitKey(0)
cv2.destroyAllWindows()

