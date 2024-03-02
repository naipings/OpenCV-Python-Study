# 应用滤波器（卷积核 或简称为 核）
# 使图像平滑（也称为图像模糊）

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 使用 5 x 5 核
kernel_averaging_5_5 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04], [0.04,
0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04],[0.04, 0.04, 0.04,
0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04]])

# 创建卷积核，方法二：
# kernel_averaging_5_5 = np.ones((5, 5), np.float32) / 25

# 使用cv2.filter2D()函数将核应用于图像：
smooth_image_f2D = cv2.filter2D(image, -1, kernel_averaging_5_5)

def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'Original image', 1)
show_with_matplotlib(smooth_image_f2D, 'cv2.filter2D()', 2)
plt.show()
