# 图像锐化
# 一种简单方法：从原始图像中减去图像的非锐化或平滑版本。

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 在以下示例中，首先应用了高斯平滑滤波器，然后从原始图像中减去生成的图像：
smoothed = cv2.GaussianBlur(image, (9, 9), 10)
unsharped = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)

# 另一种方法是：使用特定的核来锐化边缘，然后应用cv2.filter2D()函数：
kernel_sharpen_1 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

kernel_sharpen_2 = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])

kernel_sharpen_3 = np.array([[1, 1, 1],
                             [1, -7, 1],
                             [1, 1, 1]])

kernel_sharpen_4 = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 8, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 8.0

sharp_image_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
sharp_image_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
sharp_image_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
sharp_image_4 = cv2.filter2D(image, -1, kernel_sharpen_4)


def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'Original image', 1)
show_with_matplotlib(unsharped, 'sharp1', 2)
show_with_matplotlib(sharp_image_1, 'sharp2', 3)
show_with_matplotlib(sharp_image_2, 'sharp3', 1+3)
show_with_matplotlib(sharp_image_3, 'sharp4', 2+3)
show_with_matplotlib(sharp_image_4, 'sharp5', 3+3)
plt.show()