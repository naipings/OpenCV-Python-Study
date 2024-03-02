# Hu不变矩
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Hu不变矩可以保持平移、缩放和旋转不变，同时，所有的矩(第7个矩除外)对于反射都是不变的。
# 第7个矩因反射而改变，从而使其能够区分镜像图片。 
# OpenCV提供 cv2.HuMoments()函数来计算7个Hu不变矩，使用方法如下：
# cv2.HuMoments(m[, hu]) → hu

# 这里，m对应于用cv2.HuMoments()计算的矩，输出hu对应于7个Hu不变矩。
# 注：7个Hu不变矩的定义，可见：教程网址，或截屏图片。

# 接下来编写程序计算7个Hu不变矩，为了计算不变矩，必须首先使用cv2.moments()计算矩。
# 计算图像矩时，可以使用矢量形状或图像，如果 binaryImage参数为真(仅用于图像)，则输入图像中的所有非零像素将被视为1。
# 计算使用矢量形状和图像的图像矩后，根据计算的矩，计算 Hu不变矩。

# 加载图像并将其转化为灰度图像
image = cv2.imread("../opencvStudy/test43/imgs/test03.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 获取二值图像
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)

# 计算图像矩，传递参数为图像
M = cv2.moments(thresh, True)
print("moments: '{}'".format(M))

def centroid(moments):
    """根据图像矩计算质心"""
    x_centroid = round(moments['m10'] / moments['m00'])
    y_centroid = round(moments['m01'] / moments['m00'])
    return x_centroid, y_centroid

# 计算质心
x, y = centroid(M)

# 计算 Hu 矩并打印
HuM = cv2.HuMoments(M)
print("Hu moments: '{}'".format(HuM))

# 计算图像矩时传递轮廓，重复以上计算过程
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
M2 = cv2.moments(contours[0])
print("moments: '{}'".format(M2))
x2, y2 = centroid(M2)

# 绘制轮廓
# 查看轮廓外观（详见test38.py）
def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

draw_contour_outline(image, contours, (255, 0, 0), 10)

# 绘制质心
cv2.circle(image, (x, y), 25, (255, 255, 0), -1)
cv2.circle(image, (x2, y2), 25, (0, 0, 255), -1)

# 打印质心
print("('x','y'): ('{}','{}')".format(x, y))
print("('x2','y2'): ('{}','{}')".format(x2, y2))

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def show_thresh_with_matplotlib(thresh, title, pos):
    ax = plt.subplot(1, 2, pos)
    plt.imshow(thresh, cmap='gray')
    plt.title(title, fontsize=8)
    plt.axis('off')

show_img_with_matplotlib(image, "detected contour and centroid", 1)
show_thresh_with_matplotlib(thresh, 'thresh', 2)

plt.show()

# 观察所计算的矩，Hu不变矩，以及质心，可以发现使用矢量形状和图像的结果是相似的，但稍有不同，
# 例如，获得的质心：
# ('x','y'): ('3424','1900')
# ('x2','y2'): ('1087','2876')

# 其坐标相差一些像素，原因是栅格化的图像分辨率有限。
# 对于轮廓估计的矩与针对栅格化后的轮廓计算的矩稍有不同。
# 上图中可以看到程序的输出，其中使用不同颜色显示了两个质心，以便观察它们之前的差异。