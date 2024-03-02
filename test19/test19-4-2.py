# 对真实图像的按位运算
# 需要注意的是，加载的真实图像应该具有相同的形状：

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test19/imgs/test03.jpg')
binary_image = cv2.imread('../opencvStudy/test19/imgs/test04.png')

# debug
# print(image.shape, binary_image.shape)
# print(image.dtype, binary_image.dtype)

bitwise_and = cv2.bitwise_and(image, binary_image)
bitwise_or = cv2.bitwise_or(image, binary_image)
bitwise_xor = cv2.bitwise_xor(image, binary_image)

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
plt.suptitle("Bitwise AND/OR between two images", fontsize=14, fontweight='bold')
show_with_matplotlib(image, "image", 1)
show_with_matplotlib(bitwise_and, "AND image", 2)
show_with_matplotlib(bitwise_or, "OR operation", 1+2)
show_with_matplotlib(bitwise_xor, "XOR operation", 2+2)
# Show the Figure:
plt.show()