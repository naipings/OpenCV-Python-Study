# 图像平滑
# 图像平滑技术通常用于减少噪声，此外，这些技术还用于减少低分辨率图像中的像素化。

# 1.均值滤波

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
kernel_averaging_10_10 = np.ones((10, 10), np.float32) / 25

# 使用cv2.filter2D()函数将核应用于图像：
smooth_image_f2D_1 = cv2.filter2D(image, -1, kernel_averaging_5_5)
smooth_image_f2D_2 = cv2.filter2D(image, -1, kernel_averaging_5_5)

# 以下两行代码是等价的
smooth_image_b = cv2.blur(image, (10, 10))
smooth_image_bfi = cv2.boxFilter(image, -1, (10, 10), normalize=True)

def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'original', 1)
show_with_matplotlib(smooth_image_f2D_1, 'cv2.filter2D()(5,5)kernel', 2)
show_with_matplotlib(smooth_image_f2D_2, 'cv2.filter2D(10,10)kernel', 3)
show_with_matplotlib(image, 'original', 1+3)
show_with_matplotlib(smooth_image_b, 'cv2.blur()', 2+3)
show_with_matplotlib(smooth_image_bfi, 'cv2.boxFilter()', 3+3)
plt.show()