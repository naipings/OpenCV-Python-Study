# 颜色直方图
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 在多通道图像中，计算颜色直方图的实质是：计算每个通道的直方图，
# 因此我们需要创建函数来计算多个通道的直方图：
def hist_color_img(img):
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr
# 我们也可以创建一个for循环或类似的方法来调用cv2.calcHist()函数三次。

# 接下来需要调用hist_color_img()计算图像的颜色直方图：
# 加载图像
# image = cv2.imread('../opencvStudy/test26/imgs/test01.jpeg')
image = cv2.imread('../opencvStudy/test26/imgs/test2.jpg')

# 计算图像的颜色直方图
hist_color = hist_color_img(image)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

show_img_with_matplotlib(image, "image", 1)

# 可视化颜色直方图函数
def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)

show_hist_with_matplotlib_rgb(hist_color, "color histogram", 4, ['b', 'g', 'r'])

# 同样，使用cv2.add()和cv2.subtract()来修改加载的BGR图像的亮度（原始BGR图像的每个像素值添加/减去10），
# 并查看直方图的变化情况：
M = np.ones(image.shape, dtype="uint8") * 10
# 原始 BGR 图像的每个像素值添加 10
added_image = cv2.add(image, M)
hist_color_added_image = hist_color_img(added_image)

# 原始 BGR 图像的每个像素值减去 10
subtracted_image = cv2.subtract(image, M)
hist_color_subtracted_image = hist_color_img(subtracted_image)

# 可视化
show_img_with_matplotlib(added_image, "image lighter", 2)
show_hist_with_matplotlib_rgb(hist_color_added_image, "color histogram", 5, ['b', 'g', 'r'])
show_img_with_matplotlib(subtracted_image, "image darker", 3)
show_hist_with_matplotlib_rgb(hist_color_subtracted_image, "color histogram", 6, ['b', 'g', 'r'])

plt.show()