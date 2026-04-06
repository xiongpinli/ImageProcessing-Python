# -*- coding: utf-8 -*-
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
 
#读取图像
img = cv2.imread('Lena.png')
lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#Roberts算子 (罗伯特算子）
kernelx = np.array([[-1,0],[0,1]], dtype=int)
kernely = np.array([[0,-1],[1,0]], dtype=int)

'''
filter2D = 图像卷积
就是用一个 小卷积核（kernel） 在图像上从左到右、从上到下滑动，每到一个位置：
窗口内像素 × 对应核数值 → 全部相加 → 结果作为中心像素新值。
'''
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

'''
做边缘检测（Roberts、Sobel、Prewitt、Laplacian）时，计算出来的梯度值会：有负数,可能大于 255
图像像素只能是：0 ~ 255 的整数
所以必须用 convertScaleAbs 做三件事：
取绝对值（去掉负号）
缩放（把数值缩到 0~255 范围）
转成 8 位无符号整数
'''
absX = cv2.convertScaleAbs(x)      
absY = cv2.convertScaleAbs(y)

'''
dst = src1 × alpha + src2 × beta + gamma
把水平、垂直边缘加权合成一张图
'''
Roberts = cv2.addWeighted(absX,0.5,absY,0.5,0)

# ==================== Roberts 锐化 ====================
# 锐化公式：原图 + 边缘
roberts_sharp = cv2.add(grayImage, Roberts)

plt.show()
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#显示图形
titles = [u'原始图像',u'原始灰度图', u'Roberts算子',u'Roberts算子+锐化']
images = [lenna_img, grayImage,Roberts,roberts_sharp]
for i in range(4):
   plt.subplot(1,4,i+1), plt.imshow(images[i], 'gray')
   plt.title(titles[i])  
   plt.xticks([])
   plt.yticks([])
plt.show()
