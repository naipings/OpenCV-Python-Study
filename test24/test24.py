# 直方图简介

# 参见：https://blog.csdn.net/LOVEmy134611/article/details/120069404

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 为了更好的理解直方图，我们构建一个由7个不同的灰度级方块组成的图形。
# 灰度值分别为：30、60、90、120、150、180和210.
def build_sample_image():
    # 定义不同的灰度值: 60, 90, 120, ..., 210
    tones = np.arange(start=60, stop=240, step=30)
    # 使用灰度值30初始化第一个60x60方块
    result = np.ones((60, 60, 3), dtype="uint8") * 30
    # 连接构建的所有灰度方块
    for tone in tones:
        img = np.ones((60, 60, 3), dtype="uint8") * tone
        result = np.concatenate((result, img), axis=1)

    return result

def build_sample_image_2():
    # 翻转构建的灰度图像
    # img = np.fliplr(build_sample_image())
    img = np.fliplr(build_sample_image_3())
    return img

# 加载真实图像
def build_sample_image_3():
    image = cv2.imread('../opencvStudy/test24/imgs/test2.jpg')
    return image

# 接下来，构建直方图并进行可视化：
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(2, 2, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(14, 10))
plt.suptitle("Grayscale histograms introduction", fontsize=14, fontweight='bold')

# 构建图像并转换为灰度图像
# image = build_sample_image()
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image_2 = build_sample_image_2()
# gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# 使用真实图像
image = build_sample_image_3()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_2 = build_sample_image_2()
gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# 构建直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_2 = cv2.calcHist([gray_image_2], [0], None, [256], [0, 256])

# 绘制灰度图像及直方图
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "image with 60x60 regions of different tones of gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 2, 'm')
show_img_with_matplotlib(cv2.cvtColor(gray_image_2, cv2.COLOR_GRAY2BGR), "image with 60x60 regions of different tones of gray", 3)
show_hist_with_matplotlib_gray(hist_2, "grayscale histogram", 4, 'm')

plt.show()

# 直方图（图中右侧的图形）显示图像中每个色调值出现的次数（频率），
# 由于每个方块区域的大小为60*60=3600，因此上述所有灰度值的频率都为3600，其他值为0

# 由于直方图仅显示统计信息，而不显示像素的位置。所以也可以加载真实图片，并h绘制直方图，只需要修改build_sample_image()函数：
# def build_sample_image():
#     image = cv2.imread('../opencvStudy/test24/imgs/test2.jpg')
#     return image


# 补充：直方图相关术语，参见网址里面内容。
