# 颜色直方图均衡化
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 使用相同的方法，我们可以在彩色图像中执行直方图均衡，
# 将直方图均衡应用于BGR图像的每个通道（虽然这不是彩色图像直方图均衡的最佳方法），
# 创建equalize_hist_color()函数，使用cv2.split()分割BGR图像并将cv2.equalizeHist()函数应用于每个通道，
# 最后，使用cv2.merge()合并结果通道：
def equalize_hist_color(img):
    # 使用 cv2.split() 分割 BGR 图像
    channels = cv2.split(img)
    eq_channels = []
    # 将 cv2.equalizeHist() 函数应用于每个通道
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    # 使用 cv2.merge() 合并所有结果通道
    eq_image = cv2.merge(eq_channels)
    return eq_image

# 接下来，将此函数应用于三个不同的图像：原始BGR图像、将原始图像的每个像素值+10、将原始图像的每个像素值-10，并计算直方图（均衡前后的直方图）：
# 加载图像
image = cv2.imread('../opencvStudy/test28/imgs/test2.jpg')
# （注：hist_color_img()相关知识见：test26）
def hist_color_img(img):
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr
# 计算直方图均衡前后的直方图
hist_color = hist_color_img(image)
image_eq = equalize_hist_color(image)
hist_image_eq = hist_color_img(image_eq)

M = np.ones(image.shape, dtype="uint8") * 10
# 为图像的每个像素添加 10
added_image = cv2.add(image, M)
# 直方图均衡前后的直方图
hist_color_added_image = hist_color_img(added_image)
added_image_eq = equalize_hist_color(added_image)
hist_added_image_eq = hist_color_img(added_image_eq)
# 为图像的每个像素减去 10
subtracted_image = cv2.subtract(image, M)
# 直方图均衡前后的直方图
hist_color_subtracted_image = hist_color_img(subtracted_image)
subtracted_image_eq = equalize_hist_color(subtracted_image)
hist_subtracted_image_eq = hist_color_img(subtracted_image_eq)

# 最后，绘制所有这些图像：
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    ax = plt.subplot(3, 4, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)
# 可视化
show_img_with_matplotlib(image, "image", 1)
show_hist_with_matplotlib_rgb(hist_color, "color histogram", 2, ['b', 'g', 'r'])
show_img_with_matplotlib(added_image, "image lighter", 5)
show_hist_with_matplotlib_rgb(hist_color_added_image, "color histogram", 6, ['b', 'g', 'r'])
show_img_with_matplotlib(subtracted_image, "image darker", 9)
show_hist_with_matplotlib_rgb(hist_color_subtracted_image, "color histogram", 10, ['b', 'g', 'r'])
# 其他图像的可视化方法类似，不再赘述
show_img_with_matplotlib(image_eq, "image equalized", 3)
show_hist_with_matplotlib_rgb(hist_image_eq, "grayscale histogram", 4, ['b', 'g', 'r'])
show_img_with_matplotlib(added_image_eq, "image lighter equalized", 7)
show_hist_with_matplotlib_rgb(hist_added_image_eq, "grayscale histogram", 8, ['b', 'g', 'r'])
show_img_with_matplotlib(subtracted_image_eq, "image darker equalized", 11)
show_hist_with_matplotlib_rgb(hist_subtracted_image_eq, "grayscale histogram", 12, ['b', 'g', 'r'])

plt.show()

# 将直方图均衡化应用于BGR图像的每个通道并不是颜色直方图均衡化的好方法，
# 这是由于BGR色彩空间的加性特性导致彩色图像的颜色变化很大。
# 由于我们独立地改变三个通道中的亮度和对比度，因此在合并均衡通道时，这可能会导致图像中出现新的色调，正如上图所看到的那样。
# 一种颜色直方图均衡化更好的方法：见test28-2-2.py

