# 轮廓介绍
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 参见：https://blog.csdn.net/LOVEmy134611/article/details/120865039

# 轮廓视为对象边界曲线包含的所有点，通过对这些点的分析可以进行形状判断以及对象检测和识别等计算机视觉过程。
# OpenCV提供了许多函数来检测和处理轮廓，在深入了解这些函数之前，我们首先通过函数模拟观察轮廓的基本结构：
def get_test_contour():
    cnts = [np.array(
        [[[600, 320]], [[460, 562]], [[180, 563]], [[40, 320]], 
         [[179, 78]], [[459, 77]]], dtype=np.int32)]
    return cnts

# 如上所示，轮廓是由np.int32类型的多个点组成的数组，调用此函数可以获取此阵列表示的轮廓，此阵列只有包含一个轮廓：
contours = get_test_contour()
print("contour shape: '{}'".format(contours[0].shape))
print("'detected' contours: '{}' ".format(len(contours)))

# 获得轮廓后，我们可以应用OpenCV提供与轮廓相关的所有函数。
# 请注意，get_one_contour()函数中仅包含简单轮廓，而在实际场景中，检测到的真实轮廓通常有数百个点，因此调试代码将十分耗时，
# 此时设置一个简单轮廓(例如此处的 get_one_contour() 函数)以调试和测试与轮廓相关的函数将非常有用。

# OpenCV提供了cv2.drawContours()函数用于在图像中绘制轮廓，我们可以调用此函数来查看轮廓外观：
def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

# 此外，我们可能还想要绘制图像中的轮廓点：
def draw_contour_points(img, cnts, color):
    for cnt in cnts:
        # 维度压缩
        squeeze = np.squeeze(cnt)
        # 遍历轮廓阵列的所有点
        for p in squeeze:
            # 为了绘制圆点，需要将列表转换为圆心元组
            p = array_to_tuple(p)
            # 绘制轮廓点
            cv2.circle(img, p, 10, color, -1)

    return img
    
def array_to_tuple(arr):
    """将列表转换为元组"""
    return tuple(arr.reshape(1, -1)[0])

# 最后，调用draw_contour_outline() 和draw_contour_points()函数绘制轮廓和轮廓点，并可视化：
# 创建画布并复制，用于显示不同检测效果
canvas = np.zeros((640, 640, 3), dtype="uint8")
image_contour_points = canvas.copy()
image_contour_outline = canvas.copy()
image_contour_points_outline = canvas.copy()
# 绘制轮轮廓点
draw_contour_points(image_contour_points, contours, (255, 0, 255))

# 绘制轮廓
draw_contour_outline(image_contour_outline, contours, (0, 255, 255), 3)

# 同时绘制轮廓和轮廓点
draw_contour_outline(image_contour_points_outline, contours, (255, 0, 0), 3)
draw_contour_points(image_contour_points_outline, contours, (0, 0, 255))

# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    
# 绘制图像
show_img_with_matplotlib(image_contour_points, "contour points", 1)
show_img_with_matplotlib(image_contour_outline, "contour outline", 2)
show_img_with_matplotlib(image_contour_points_outline, "contour outline and points", 3)

# 可视化
plt.show()
