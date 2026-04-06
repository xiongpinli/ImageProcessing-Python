# -*- coding: utf-8 -*-

'''
对噪声比 Roberts 稳
边缘定位中等
常用于简单场景的边缘检测
'''
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
 
#读取图像
img = cv2.imread('lena.png')
lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
'''
第一步： Prewitt 是用 3×3 模板分别算水平、垂直边缘，最后加权融合
'''
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)

'''
c
filter2D 原图 × Gx → 得到水平梯度
filter2D 原图 × Gy → 得到垂直梯度
'''
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

'''
第三步：
convertScaleAbs 对 Gx、Gy 取绝对值
'''
absX = cv2.convertScaleAbs(x)       
absY = cv2.convertScaleAbs(y)

'''
第四步
addWeighted 融合 Gx + Gy → 最终边缘图
'''
Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图形
titles = [u'原始图像', u'Prewitt算子']  
images = [lenna_img, Prewitt]  
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
