# 比较CLAHE和直方图均衡化
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 为了完整起见，接下来对比CLAHE和直方图均衡化（cv2.equalizeHist()）在同一图像上的效果，同时可视化生成的图像和生成的直方图。
image = cv2.imread('../opencvStudy/test30/imgs/test01.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# 直方图均衡化
gray_image_eq = cv2.equalizeHist(gray_image)
# 计算直方图
hist_eq = cv2.calcHist([gray_image_eq], [0], None, [256], [0, 256])
# 创建 clahe:
clahe = cv2.createCLAHE(clipLimit=4.0)
# 在灰度图像上应用 clahe
gray_image_clahe = clahe.apply(gray_image)
# 计算直方图
hist_clahe = cv2.calcHist([gray_image_clahe], [0], None, [256], [0, 256])

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 2)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 3)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 4, 'm')
show_hist_with_matplotlib_gray(hist_eq, "grayscale histogram", 5, 'm')
show_hist_with_matplotlib_gray(hist_clahe, "grayscale histogram", 6, 'm')

plt.show()

# 通过结果对比可知，可以肯定地说，在许多情况下，CLAHE比应用直方图均衡化有更好的结果和性能。