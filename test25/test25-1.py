# OpenCV中使用cv2.calcHist()函数来计算一个或多个的直方图。因此，该函数可以应用于单通道图像和多通道图像。
# 计算灰度图像的直方图：
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# 参数详情：见截屏。
# 具体参见：https://blog.csdn.net/LOVEmy134611/article/details/120069404


# 不带蒙版的灰度直方图

# 计算全灰度图像（无蒙版）直方图的代码如下：
# image = cv2.imread('example.png')
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
# show_hist_with_matplotlib_gray(hist, "grayscale histogram", 4, 'm')
# 在上述代码中，hist是一个形状为（256，1）的数组，每个值（bin）对应于具有相应色调值的像素数。

# 我们可以对灰度图像执行图像加法或减法，以便修改图像中每个像素的灰度强度，以观察如何更改图像的亮度以及直方图如何变化：
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

gray_image = cv2.imread('../opencvStudy/test25/imgs/test2.jpg', cv2.IMREAD_GRAYSCALE)

M = np.ones(gray_image.shape, dtype="uint8") * 30

# 计算原灰度图像直方图
hist_gray_image = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# 每个灰度值加 30
added_image = cv2.add(gray_image, M)

# 计算结果图像的直方图
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

# 每个灰度值减 30
subtracted_image = cv2.subtract(gray_image, M)

# 计算结果图像的直方图
hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(14, 10))
plt.suptitle("Grayscale histograms introduction", fontsize=14, fontweight='bold')

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist_gray_image, "grayscale histogram", 4, 'm')

show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 2)
show_hist_with_matplotlib_gray(hist_added_image, "grayscale histogram", 5, 'm')

show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "gray darker", 3)
show_hist_with_matplotlib_gray(hist_subtracted_image, "grayscale histogram", 6, 'm')

plt.show()

# 中间的灰度图像对应于原始图像的每个像素+30的图像，从而产生更亮的图像，图像的直方图会向右偏移，因为没有强度在[0-30]范围内的像素；
# 而右侧的灰度图像对应于原始图像的每个像素都-30的图像，从而导致图像更暗，图像的直方图会向左偏移，因为没有强度在[225-255]范围内的像素。
