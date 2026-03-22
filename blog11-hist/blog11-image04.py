#encoding:utf-8
import cv2  
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread('Lena.png',cv2.IMREAD_GRAYSCALE)
his = cv2.calcHist(src,[0],None,[256],[0,256])
cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.plot(his, color='b')
plt.show()
