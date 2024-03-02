# 颜色直方图均衡化
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 一种颜色直方图均衡化更好的方法：是将BGR图像转换为包含亮度/强度通道的色彩空间（Yuv、Lab、HSV、HSL）。
# 然后，只在亮度通道上应用直方图均衡，最后合并通道并将它们转换回BGR颜色空间，
# 以HSV为例，创建equalize_hist_color_hsv()函数实现上述颜色直方图归一化方法：
def equalize_hist_color_hsv(img):
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
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

hist_color = hist_color_img(image)
# 计算直方图均衡前后的直方图
image_eq = equalize_hist_color_hsv(image)
hist_image_eq = hist_color_img(image_eq)

M = np.ones(image.shape, dtype="uint8") * 10
# 为图像的每个像素添加 10
added_image = cv2.add(image, M)
hist_color_added_image = hist_color_img(added_image)
# 直方图均衡前后的直方图
added_image_eq = equalize_hist_color_hsv(added_image)
hist_added_image_eq = hist_color_img(added_image_eq)
# 为图像的每个像素减去 10
subtracted_image = cv2.subtract(image, M)
hist_color_subtracted_image = hist_color_img(subtracted_image)
# 直方图均衡前后的直方图
subtracted_image_eq = equalize_hist_color_hsv(subtracted_image)
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

# 由上图可以看出，仅均衡HSV图像的V通道得到的结果比均衡BGR图像的所有通道的效果要好很多，
# 也可以将这种方法用于其他包含亮度/强度通道的色彩空间（Yuv、Lab和HSL）。