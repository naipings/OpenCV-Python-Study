# 轮廓压缩
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 当检测到的轮廓包含大量点时，可以使用轮廓算法来减少轮廓点的数量，OpenCV提供了减少轮廓点数量的方法，这就是cv2.findContourd()函数中method参数的用武之地了：（具体可见：教程网址，或者截屏图片）

# 绘制一些预定义的形状（见test39.py）
def build_sample_image():
    """绘制一些基本形状"""
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 255), -1)
    cv2.rectangle(img, (100, 100), (200, 200), (70, 70, 70), -1)
    cv2.circle(img, (350, 350), 100, (255, 255, 0), -1)
    cv2.circle(img, (350, 350), 50, (70, 70, 70), -1)
    return img

# 加载图像并转换为灰度图像
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 阈值处理
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

methods = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS]

# 通过函数模拟观察轮廓的基本结构：（见test38.py）
def get_test_contour():
    cnts = [np.array(
        [[[600, 320]], [[460, 562]], [[180, 563]], [[40, 320]], 
         [[179, 78]], [[459, 77]]], dtype=np.int32)]
    return cnts

# 绘制图像中的轮廓点：（见test38.py）
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

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    

# 循环使用每一压缩算法来比较它们之间的区别
for index in range(len(methods)):
    method = methods[index]
    image_approx = image.copy()
    contours ,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, method)
    # 可视化
    draw_contour_points(image_approx, contours, (255, 255, 255))
    show_img_with_matplotlib(image_approx, "contours ({})".format(method), 3 + index)

show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)

plt.show()
