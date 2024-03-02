# 高斯滤波

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# (9, 9)表示高斯矩阵的长与宽都是5，标准差取0
smooth_image_gb = cv2.GaussianBlur(image, (9, 9), 0)
# 标准差取0.3
smooth_image_gb_2 = cv2.GaussianBlur(image, (9, 9), 0.3)
# 构建高斯核
print(cv2.getGaussianKernel(9,0))

def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'original', 1)
show_with_matplotlib(smooth_image_gb, 'cv2.GaussianBlur()', 2)
show_with_matplotlib(smooth_image_gb_2, 'cv2.GaussianBlur()', 3)
plt.show()