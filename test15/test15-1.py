# 缩放变换

# 法1：
# 指定缩放后图像尺寸
# resized_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

# 法1：
# 提供缩放因子fx和fy的值。例如，如果要将图像缩小2倍：
# 使用缩放因子
# dst_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# 关于cv2.resize()函数采用的插值方法，
# 具体参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188


import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 获取图像尺寸
height, width, channels = image.shape

dst_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
dst_image_2 = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# 显示缩放后的图像
def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'Original image', 1)
show_with_matplotlib(dst_image, 'Resized image', 2)
show_with_matplotlib(dst_image_2, 'Resized image 2', 3)
plt.show()

# 可以通过坐标系观察图片的缩放情况。