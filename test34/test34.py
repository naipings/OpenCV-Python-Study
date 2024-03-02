# 自适应阈值算法
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 当由于图像不同区域的光照条件不同时，为了获取更好的阈值结果，可以尝试使用自适应阈值。
# 在 OpenCV 中，自适应阈值由 cv2.adapativeThreshold() 函数实现，其用法如下：
# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst

# 此函数将自适应阈值应用于src数组(8位单通道图像)。
# maxValue参数用于设置dst图像中满足条件的像素值。
# adaptiveMethod参数设置要使用的自适应阈值算法，有以下两个可选值：
# 1.cv2.ADAPTIVE_THRESH_MEAN_C：T(x, y) 阈值为 (x, y) 的 blockSize x blockSize 邻域的平均值减去参数C
# 2.cv2.ADAPTIVE_THRESH_GAUSSIAN_C：T(x, y) 阈值为 (x, y) 的blockSize x blockSize 邻域的加权均值减去参数C

# blockSize参数用于计算像素阈值的邻域区域的大小，
# C参数是均值或加权均值需要减去的常数(取决于由 adaptiveMethod 参数设置的自适应方法)，
# thresholdType参数可选值包括 cv2.THRESH_BINARY 和 cv2.THRESH_BINARY_INV。

# 综上所属，cv2.adapativeThreshold() 函数的计算公式如下，其中 T(x, y) 是为每个像素计算的阈值：
# 公式见：https://blog.csdn.net/LOVEmy134611/article/details/120069509

# 应用不同自适应阈值算法，进行图像处理：
# 加载图像
image = cv2.imread('../opencvStudy/test34/imgs/test01.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 应用自适应阈值处理
thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Adaptive thresholding", fontsize=14, fontweight='bold')

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=11, C=2", 2)
# 其他图像可视化方法类似，不再赘述
show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=31, C=3", 3)
show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=11, C=2", 5)
show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=31, C=3", 6)

plt.show()

# 在上图中，可以看到应用具有不同参数的cv2.adaptiveThreshold()处理后的输出，自适应阈值可以提供更好的阈值图像。
# 但是，图像中也会出现了很多噪点。为了减少噪点，可以应用一些平滑操作，例如双边滤波，因为它会在去除噪声的同时保持锐利边缘。
# 因此，我们在对图像进行阈值处理之前首先应用 OpenCV 提供的双边滤波函数 cv2.bilateralFilter()：
# 见test34-2.py