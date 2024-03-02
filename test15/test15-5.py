# 图像的透视变换

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 获取图像尺寸
height, width = image.shape[:2]

# 使用cv2.getPerspectiveTransform()函数构建3*3变换矩阵。
# 该函数需要4对点（源图像和输出图像中四边形的坐标），函数会根据这些点计算透视变换矩阵。
# 然后，将M矩阵传递给cv2.warpPerspective()：
pts_1 = np.float32([[2402, 496], [2857, 430], [2561, 1000], [2930, 943]])
pts_2 = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])
M = cv2.getPerspectiveTransform(pts_1, pts_2)
dst_image = cv2.warpPerspective(image, M, (800, 800))

# 显示透视变换后的图像
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