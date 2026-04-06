#encoding:utf-8
import cv2  
import numpy as np
import matplotlib.pyplot as plt
 
#读取图片
image = cv2.imread('Lena.png')
#image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#图像平移矩阵
M = np.float32([[1, 0, 80], [0, 1, 30]])
rows, cols = image.shape[:2]
img1 = cv2.warpAffine(image, M, (cols, rows))
cv2.imshow("平移前-image", image)
cv2.imshow('平移后-image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


#绕图像的中心旋转
#源图像的高、宽 以及通道数
rows, cols, channel = image.shape
# 参数	说明
# center	旋转中心点坐标 (x, y)，通常设为图像的中心点 (cols/2, rows/2)
# angle	旋转角度（单位：度）。正值表示逆时针旋转，负值表示顺时针旋转
# scale	图像缩放因子。1.0 表示不缩放，小于 1 表示缩小，大于 1 表示放大

M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 2)
print("-----------生成二维旋转仿射变换矩阵------")
print(M)

#函数参数：原始图像 旋转参数 元素图像宽高
img2 = cv2.warpAffine(image, M, (cols, rows))
cv2.imshow("Rotation-before", image)
cv2.imshow('Rotation-after', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#图像翻转
img3 = cv2.flip(image, 0)  # 0 垂直翻转
img4 = cv2.flip(image, 1)   # 水平翻转
img41 = cv2.flip(image, -1)   # 水平翻转
cv2.imshow("flip", image)
cv2.imshow("flip-vertical", img3)
cv2.imshow('flip-horizontal', img4)
cv2.imshow("flip-d", img41)
cv2.waitKey(0)
cv2.destroyAllWindows()


#图像缩小
print(image.shape)
img5 = cv2.resize(image, (200,100))
#图像放大
img6= cv2.resize(image, None, fx=2, fy=3)
cv2.imshow("resize-before", image)
cv2.imshow("resize-1", img5)
cv2.imshow('resize-2', img6)
cv2.waitKey(0)
cv2.destroyAllWindows()


#图像的仿射
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
img7 = cv2.warpAffine(image, M, (rows,cols))
cv2.imshow("AffineTransform", img7)
cv2.waitKey(0)
cv2.destroyAllWindows()
#图像的透射
pts1 = np.float32([[56,65],[238,52],[28,237],[239,240]])
pts2 = np.float32([[0,0],[200,0],[0,200],[200,200]])
M = cv2.getPerspectiveTransform(pts1,pts2)
img8 = cv2.warpPerspective(image,M,(200,200))
cv2.imshow("PerspectiveTransform", img8)
cv2.waitKey(0)
cv2.destroyAllWindows()

# #循环显示图形
# titles = [ 'source', 'shift', 'reduction', 'enlarge', 'rotation', 'flipX', 'flipY', 'affine', 'transmission']
# images = [image, img1, img2, img3, img4, img5, img6, img7, img8]
# for i in range(9):
#    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
# plt.show()
