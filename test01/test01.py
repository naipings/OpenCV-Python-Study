# OpenCV通道顺序

import cv2
# cv2.imread()函数以BGR顺序加载图像
img_OpebCV = cv2.imread('../opencvStudy/test01/imgs/test01.jpeg')
# 然后我们使用cv2.split将加载的图像分成三个通道(b, g, r)
[b, g, r] = cv2.split(img_OpebCV)
# 我们更改b和r通道的顺序以遵循RGB格式，即我们所需要的Matplotlib格式
img_matplotlib = cv2.merge([r, g, b])
# 显示图像：
# 1.使用plt.imshow
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.subplot(121)
plt.imshow(img_OpebCV)
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.show()

# 2.使用cv2.imshow
# cv2.imshow('bgr image', cv2.resize(img_OpebCV,None,fx=0.1,fy=0.1))
# cv2.imshow('rgb image', cv2.resize(img_matplotlib,None,fx=0.1,fy=0.1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# img_concats = np.concatenate((img_OpebCV, img_matplotlib),axis=1)
# cv2.imshow('bgr image and rgb image', cv2.resize(img_concats,None,fx=0.1,fy=0.1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 获取图像的一个通道
# B = img_OpebCV[:, :, 0]
# G = img_OpebCV[:, :, 1]
# R = img_OpebCV[:, :, 2]

# 可以使用Numpy在一条语句中将图像从BGR转换为RGB
# img_matplotlib = img_OpebCV[:, :, -1]
