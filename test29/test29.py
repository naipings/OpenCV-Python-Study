# 对比度受限的自适应直方图均衡化
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 背景简介，参见：https://blog.csdn.net/LOVEmy134611/article/details/120204604

# 将CLAHE应用于灰度和彩色图像。应用CLAHE时，有两个重要参数，
# 第一个是clipLimit，它设置对比度限制的阈值，默认值为10；
# 第二个是tileGridSize，它设置行和列中的tiles数量。
# 应用CLAHE时，图像被划分成为tiles(默认为8*8)的小块以执行其计算。

# 将CLAHE应用于灰度图像，需要使用以下代码：
# 加载图像
image = cv2.imread('../opencvStudy/test29/imgs/test01.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 灰度图像应用 CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0)
gray_image_clahe = clahe.apply(gray_image)
# 使用不同 clipLimit 值
clahe.setClipLimit(5.0)
gray_image_clahe_2 = clahe.apply(gray_image)
clahe.setClipLimit(10.0)
gray_image_clahe_3 = clahe.apply(gray_image)
clahe.setClipLimit(20.0)
gray_image_clahe_4 = clahe.apply(gray_image)

# 然后，我们将CLAHE应用于彩色图像，类似于彩色图像对比度均衡的方法，
# 创建四个函数以仅在不同颜色空间的亮度通道上使用CLAHE来均衡化彩色图像：
def equalize_clahe_color_hsv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cla.apply(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image

def equalize_clahe_color_lab(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    eq_L = cla.apply(L)
    eq_image = cv2.cvtColor(cv2.merge([eq_L, a, b]), cv2.COLOR_Lab2BGR)
    return eq_image

def equalize_clahe_color_yuv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    Y, U, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eq_Y = cla.apply(Y)
    eq_image = cv2.cvtColor(cv2.merge([eq_Y, U, V]), cv2.COLOR_YUV2BGR)
    return eq_image

def equalize_clahe_color(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cla.apply(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image
# 彩色图像应用 CLAHE
image_clahe_color = equalize_clahe_color(image)
image_clahe_color_lab = equalize_clahe_color_lab(image)
image_clahe_color_hsv = equalize_clahe_color_hsv(image)
image_clahe_color_yuv = equalize_clahe_color_yuv(image)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 5, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    ax = plt.subplot(2, 5, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=2.0", 2)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_2, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=5.0", 3)
# 其他图像的可视化方法类似，不再赘述
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_3, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=10.0", 4)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_4, cv2.COLOR_GRAY2BGR), "gray CLAHE clipLimit=20.0", 5)

show_img_with_matplotlib(image, "color", 6)
show_img_with_matplotlib(image_clahe_color, "clahe on each channel(BGR)", 7)
show_img_with_matplotlib(image_clahe_color_lab, "clahe on L channel(LAB)", 8)
show_img_with_matplotlib(image_clahe_color_hsv, "clahe on V channel(HSV)", 9)
show_img_with_matplotlib(image_clahe_color_yuv, "clahe on Y channel(YUV)", 10)

plt.show()
