# OpenCV中的色彩映射

# 为了执行这种色彩转换，OpenCV包含多种色彩映射来增强可视化效果，cv2.applyColorMap()函数在给定的图像上应用色彩映射，
# 例如应用cv2.COLORMAP_HSV色彩映射：
# img_COLORMAP_HSV = cv2.applyColorMap(gray_img, cv2.COLORMAP_HSV)

# OpenCV定义的色彩映射如下，
# 参见：https://blog.csdn.net/LOVEmy134611/article/details/120069317

# 我们可以将所有的颜色映射应用到同一个灰度图像上，并将它们绘制在同一个图形中：
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 7, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 加载图像并转化为灰度图像
gray_img = cv2.imread('../opencvStudy/test21/imgs/test2.jpg', cv2.IMREAD_GRAYSCALE)

# 色彩映射列表
colormaps = ["AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING", "COOL", "HSV", "HOT", "PINK", "PARULA"]

plt.figure(figsize=(12, 5))
plt.suptitle("Colormaps", fontsize=14, fontweight='bold')

show_with_matplotlib(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), "GRAY", 1)

# 应用色彩映射
for idx, val in enumerate(colormaps):
    show_with_matplotlib(cv2.applyColorMap(gray_img, idx), val, idx + 2)

plt.show()

# 在上图中，可以看到将所有预定义的颜色映射应用于灰度图像以增强可视化的效果。
