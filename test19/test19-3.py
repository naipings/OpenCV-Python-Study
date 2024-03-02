# 图像混合
# 图像混合也是图像相加的一种，只是其可以赋予相加以图像不同的权重，可以得到类似透明的效果，
# 可以使用cv2.addWeighted()函数进行图像混合。

# 接下来，结合Sobel算子来观察cv2.addWeighted()函数效果。

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test19/imgs/test01.jpeg')
gray_image = cv2.imread('../opencvStudy/test19/imgs/test01.jpeg', cv2.IMREAD_GRAYSCALE)

# 输出深度设置为CV_16S，以避免溢出
# CV_16S是由2字节有符号整数（16位有符号整数）组成的通道
gradient_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, 3)
gradient_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, 3)

# 在计算处水平和垂直梯度后，可以使用函数cv2.addWeighted()将它们混合成图像，如下所示：
abs_gradient_x = cv2.convertScaleAbs(gradient_x)
abs_gradient_y = cv2.convertScaleAbs(gradient_y)
# 使用相同的权重混合两个图像
sobel_image = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)


def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
plt.figure(figsize=(10, 4))
plt.suptitle("Sobel operator and cv2.addWeighted() to show the output", fontsize=14, fontweight='bold')
show_with_matplotlib(image, "Image", 1)
show_with_matplotlib(cv2.cvtColor(abs_gradient_x, cv2.COLOR_GRAY2BGR), "Gradient x", 2)
show_with_matplotlib(cv2.cvtColor(abs_gradient_y, cv2.COLOR_GRAY2BGR), "Gradient y", 3)
show_with_matplotlib(cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2BGR), "Sobel output", 4)
# Show the Figure:
plt.show()

