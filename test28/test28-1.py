# 灰度直方图均衡化

# 参见：https://blog.csdn.net/LOVEmy134611/article/details/120204604
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 使用cv2.equalizeHist()函数来均衡给定灰度图像的对比度：
# 加载图像并转换为灰度图像
image = cv2.imread('../opencvStudy/test27/imgs/test2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# 直方图均衡化
gray_image_eq = cv2.equalizeHist(gray_image)
# 直方图均衡化后的图像直方图
hist_eq = cv2.calcHist([gray_image_eq], [0], None, [256], [0, 256])

# 为了深入了解直方图均衡，我们对原始灰度图像进行修改，为图像的每个像素添加/减去30，并计算灰度图像均衡前后的直方图
M = np.ones(gray_image.shape, dtype='uint8') * 30
# 为图像的每个像素添加 30
added_image = cv2.add(gray_image, M)
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])
# 直方图均衡化
added_image_eq = cv2.equalizeHist(gray_image_eq)
hist_eq_added_image = cv2.calcHist([added_image_eq], [0], None, [256], [0, 256])
# 为图像的每个像素减去 30
subtracted_image = cv2.subtract(gray_image, M)
hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])
# 直方图均衡化
subtracted_image_eq = cv2.equalizeHist(subtracted_image)
hist_eq_subtracted_image = cv2.calcHist([subtracted_image_eq], [0], None, [256], [0, 256])

# 最后，绘制所有这些图像：
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(3, 4, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)
# 可视化
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 2, 'm')
show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 5)
show_hist_with_matplotlib_gray(hist_added_image, "grayscale histogram", 6, 'm')
show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "gray darker", 9)
show_hist_with_matplotlib_gray(hist_subtracted_image, "grayscale histogram", 10, 'm')
# 其他图像的可视化方法类似，不再赘述（本人来进行实现）
show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "gray equalized", 3)
show_hist_with_matplotlib_gray(hist_eq, "grayscale histogram", 4, 'm')
show_img_with_matplotlib(cv2.cvtColor(added_image_eq, cv2.COLOR_GRAY2BGR), "gray lighter equalized", 7)
show_hist_with_matplotlib_gray(hist_eq_added_image, "grayscale histogram", 8, 'm')
show_img_with_matplotlib(cv2.cvtColor(subtracted_image_eq, cv2.COLOR_GRAY2BGR), "gray darker equalized", 11)
show_hist_with_matplotlib_gray(hist_eq_subtracted_image, "grayscale histogram", 12, 'm')

plt.show()

# 在上图中，我们可以看到三个均衡化后的图像非常相似，这也反映在均衡化后的直方图中，这是因为直方图均衡化倾向于标准化图像的亮度，同时增加对比度。