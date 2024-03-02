# Otsu阈值算法
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 简单阈值算法应用同一全局阈值，因此我们需要尝试不同的阈值并查看阈值图像，以满足我们的需要。
# 但是，这种方法需要大量尝试，一种改进的方法是：使用OpenCV中的cv2.adapativeThreshold()函数计算自适应阈值。
# 但是，计算自适应阈值需要两个合适的参数：blockSize 和 C；
# 另一种改进方法是：使用Otsu阈值算法，这在处理双峰图像时是非常有效(双峰图像其直方图包含两个峰值)。
# Otsu算法通过最大化两类像素之间的方差来自动计算将两个峰值分开的最佳阈值。其等效于最佳阈值最小化类内方差。
# Otsu阈值算法是一种统计方法，因为它依赖于从直方图获得的统计信息(例如，均值、方差或熵)。
# 在OpenCV中使用cv2.threshold()函数计算Otsu阈值的方法如下：
# ret, th = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 使用Otsu阈值算法，不需要设置阈值，因为Otsu阈值算法会计算最佳阈值，因此参数thresh=0，
# cv2.THRESH_OTSU标志表示将应用Otsu算法，
# 此标志可以与 cv2.THRESH_BINARY、cv2.THRESH_BINARY_INV、cv2.THRESH_TRUNC、cv2.THRESH_TOZERO 以及 cv2.THRESH_TOZERO_INV 结合使用，
# 函数返回阈值图像th和阈值ret。
# 将此算法应用于实际图像，并绘制一条线可视化阈值，确定阈值th坐标：
def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    ax = plt.subplot(2, 2, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    # 可视化阈值
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)

# 加载图像
image = cv2.imread('../opencvStudy/test35/imgs/test01.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# otsu 阈值算法
ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Otsu's binarization algorithm", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 2)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 3, 'm', ret1)
show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR), "Otsu's binarization", 4)

plt.show()

# 在上图中，可以看到源图像中没有噪声，因此算法可以正常工作，
# 接下来，我们手动为图像添加噪声，以观察噪声对Otus阈值算法的影响，
# 然后利用高斯滤波消除部分噪声，以查看阈值图像变化情况：
# 见test35-2.py