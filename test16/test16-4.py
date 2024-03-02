# 中值滤波

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 中值滤波，使用中值核 对图像进行模糊处理
smooth_image_mb = cv2.medianBlur(image, 9)
smooth_image_mb_2 = cv2.medianBlur(image, 3)

def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'original', 1)
show_with_matplotlib(smooth_image_mb, 'cv2.medianBlur()', 2)
show_with_matplotlib(smooth_image_mb_2, 'cv2.medianBlur()', 3)
plt.show()