# 裁剪图像

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 获取图像尺寸
height, width = image.shape[:2]

# 使用NumPy切片裁剪图像
dst_image = image[90:1800, 250:3200]

# 显示裁剪后的图像
def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'Original image', 1)
show_with_matplotlib(dst_image, 'Resized image', 2)
plt.show()