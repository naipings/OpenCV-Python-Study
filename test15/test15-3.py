# 旋转图像

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test15/imgs/test01.jpeg')

# 获取图像尺寸
height, width = image.shape[:2]

# 使用cv2.getRotationMatrix2D()函数构建变换矩阵。之后，将这个M矩阵应用于图像：
# 本例构建变换矩阵M，以相对于图像中心旋转180度，缩放因子为1（不缩放）。
M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 180, 1)
dst_image = cv2.warpAffine(image, M, (width, height))

# 使用不同的旋转中心进行旋转：
M = cv2.getRotationMatrix2D((width/1.5, height/1.5), 30, 1)
dst_image_2 = cv2.warpAffine(image, M, (width, height))

# 显示旋转后的图像
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

# 补充：cv2.getRotationMatrix2D()函数定义：
# retval = cv2.getRotationMatrix2D ( center,angle,scale)
#（1）center：**旋转中心点
#（2）angle：**旋转角度，正数表示逆时针旋转，负数表示顺时针旋转
#（3）scale：**变换尺度（缩放大小）