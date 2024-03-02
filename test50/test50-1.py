# 创建标记和字典
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 在本节中，我们将了解基于标记的增强现实的工作原理，我们以 ArUco算法为例讲解基于标记的增强现实。
# ArUco会自动检测标记并纠正可能的错误。此外，ArUco提出了通过将多个标记与遮挡掩码组合来解决遮挡问题的方法，遮挡掩码是通过颜色分割计算的。
# 使用标记的主要好处是可以在图像中有效且稳健基于标记执行姿势估计，其可以准确地导出标记的四个角，从先前计算的标记的四个角中获得相机姿态。

# 使用 ArUco算法的第一步是创建标记和字典。ArUco标记是由外部和内部单元(也称为位-bits)组成的方形标记。
# 外部单元格设置为黑色，从而创建一个可以快速且稳健地检测到的外部边界；内部单元格用于对标记进行编码。
# 可以创建不同尺寸的 ArUco标记，标记的尺寸表示与内部矩阵相关的内部单元格的数量。
# 例如，大小为 5 x 5 (n=5) 的标记由 25 个内部单元组成。此外，还可以设置标记边框的位数。

# 标记字典是在特定应用程序中使用的一组标记。虽然之前的算法使用固定字典，但 ArUco提出了一种自动方法来生成具有所需数量和所需位数的标记，
# 因此，ArUco包括一些预定义的字典，涵盖与标记数量和标记大小相关的配置。

# 创建基于标记的增强现实应用程序时要考虑的第一步是创建要使用的标记:
# 第一步是创建字典对象。ArUco包含了一些预定义字典：
# DICT_4X4_50 = 0
# DICT_4X4_100 = 1
# DICT_4X4_250 = 2
# DICT_4X4_1000 = 3
# DICT_5X5_50 = 4
# DICT_5X5_100 = 5
# DICT_5X5_250 = 6
# DICT_5X5_1000 = 7
# DICT_6X6_50 = 8
# DICT_6X6_100 = 9
# DICT_6X6_250 = 10
# DICT_6X6_1000 = 11
# DICT_7X7_50 = 12
# DICT_7X7_100 = 13
# DICT_7X7_250 = 14
# DICT_7X7_1000 = 15

# 使用cv2.aruco.Dictionary_get()函数创建一个由250个标记组成的字典。每个标记的大小为7*7（n=7）：
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

# 可以使用 cv2.aruco.drawMarker()函数绘制标记，该函数返回绘制的标记。 
# cv2.aruco.drawMarker()函数的
# 第一个参数是字典对象；
# 第二个参数是标记id ，范围在0到249之间，因为我们的字典有250个标记；
# 第三个参数 sidePixels是创建的标记图像的大小(以像素为单位)；
# 第四个参数是 borderBits(可选参数，默认为 1)，它设置标记边框的位数。
# 创建三个具有不同位数的标记边框：
aruco_marker_1 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=1)
aruco_marker_2 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=2)
aruco_marker_3 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=3)
# aruco_marker_1 = cv2.aruco.drawDetectedMarkers

# 将这些创建的标记进行可视化：
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=6)
    plt.axis('off')

plt.suptitle("Aruco markers creation", fontsize=14, fontweight='bold')

show_img_with_matplotlib(cv2.cvtColor(aruco_marker_1, cv2.COLOR_GRAY2BGR), "marker_DICT_7*7_250_600_1", 1)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_2, cv2.COLOR_GRAY2BGR), "marker_DICT_7*7_250_600_2", 2)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_3, cv2.COLOR_GRAY2BGR), "marker_DICT_7*7_250_600_3", 3)

plt.show()

# 运行不了可能是因为opencv-python的版本问题(要求4.7x以下Opencv版本)
# (使用本人pytorch虚拟环境，里面的opencv-python是4.5.3.56版本)
