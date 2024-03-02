# 平移图像

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 获取图像尺寸
height, width = image.shape[:2]

# 创建变换矩阵，并使用cv2.warpAffine()函数平移图像：
# x方向平移200个像素，y方向平移30像素
M = np.float32([[1, 0, 200], [0, 1, 30]])
dst_image_1 = cv2.warpAffine(image, M, (width, height))

# 反方向平移图像：
M = np.float32([[1, 0, -200], [0, 1, -30]])
dst_image_2 = cv2.warpAffine(image, M, (width, height))

# 显示平移后的图像
def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'Original image', 1)
show_with_matplotlib(dst_image_1, 'Resized image', 2)
show_with_matplotlib(dst_image_2, 'Resized image 2', 3)
plt.show()