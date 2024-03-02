# 直方图的自定义可视化
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 为了可视化直方图，我们调用了plt.plot()函数，这是由于没有OpenCV函数可以用来直接绘制直方图。
# 因此，如果我们想要使用OpenCV绘制直方图，必须利用OpenCV原语（例如：cv2.polylines() 和 cv2.rectangle() 等）来绘制直方图。

# 我们创建实现此功能的plot_hist()函数，此函数创建一个BGR彩色图像，并在其中绘制直方图。
# 该函数的代码如下：
def plot_hist(hist_items, color):
    # 出于可视化目的，我们添加了一些偏移
    offset_down = 10
    offset_up = 10
    # 这将用于创建要可视化的点（x坐标）
    x_values = np.arange(256).reshape(256, 1)
    # 创建画布
    canvas = np.ones((300, 256, 3), dtype='uint8') * 256

    for hist_item, col in zip(hist_items, color):
        # 在适当的可视化范围内进行规范化
        cv2.normalize(hist_item, hist_item, 0 + offset_down, 300 - offset_up, cv2.NORM_MINMAX)
        # 将值强制转换为int
        around = np.around(hist_item)
        # 数据类型转换
        hist = np.int32(around)
        # 使用直方图和x坐标创建点
        pts = np.column_stack((x_values, hist))
        # 绘制点
        cv2.polylines(canvas, [pts], False, col, 2)
        # 绘制一个矩形
        cv2.rectangle(canvas, (0, 0), (255, 298), (0, 0, 0), 1)
    
    # 沿上/下方向翻转图像
    res = np.flipud(canvas)
    return res

# 此函数接收直方图并为直方图的每个元素构建(x, y)点pts，其中y值表示直方图x元素的频率。
# 这些点pts是通过cv2.polylines()函数绘制的，该函数根据pts数组绘制曲线。
# 最后，图像需要垂直翻转，因为y值颠倒了。

# 最后我们使用plt.plot()和自定义函数的直方图绘制函数进行比较：
# 读取图像
image = cv2.imread('../opencvStudy/test27/imgs/test2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 plt.plot() 绘制的灰度直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# 使用 plt.plot() 绘制的颜色直方图（注：hist_color_img()相关知识见：test26）
def hist_color_img(img):
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr
hist_color = hist_color_img(image)

# 自定义灰度直方图
gray_plot = plot_hist([hist], [(255, 0, 255)])

# 自定义颜色直方图
color_plot = plot_hist(hist_color, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# 灰度直方图
def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

# 可视化颜色直方图函数
def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    ax = plt.subplot(2, 3, pos)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)

plt.figure(figsize=(14, 10))
plt.suptitle("Grayscale histograms introduction", fontsize=14, fontweight='bold')

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_img_with_matplotlib(image, "image", 4)
show_hist_with_matplotlib_gray(hist, "grayscale histogram (matplotlib)", 2, 'm')
show_hist_with_matplotlib_rgb(hist_color, "color histogram (matplotlib)", 3, ['b', 'g', 'r'])
show_img_with_matplotlib(gray_plot, "grayscale histogram (custom)", 5)
show_img_with_matplotlib(color_plot, "color histogram (custom)", 6)

plt.show()