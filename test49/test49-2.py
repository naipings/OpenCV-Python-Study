# 特征匹配
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 本节将介绍如何匹配检测到的特征。
# OpenCV提供了两种匹配器：
    # 蛮力(Brute-Force, BF)匹配器：该匹配器利用为第一组中检测到的特征计算的描述符与第二组中的所有描述符进行匹配。最后，它返回距离最近的匹配项。
    # Fast Library for Approximate Nearest Neighbors (FLANN) 匹配器：对于大型数据集，此匹配器比 BF匹配器运行速度更快，它利用了最近邻搜索的优化算法。

# 接下来，使用BF匹配器来查看如何匹配检测到的特征，第一步是检测关键点并计算描述符：
# 加载图像
image_1 = cv2.imread('../opencvStudy/test49/imgs/test16.png')
image_2 = cv2.imread('../opencvStudy/test49/imgs/test16.png')
image_2 = cv2.flip(image_2, -1) 
# ORB 检测器初始化
orb = cv2.ORB_create()
# 使用 ORB 检测关键点并计算描述符
keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)

# 下一步是使用cv2.BFMatcher()创建BF匹配器对象：
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# cv2.BFMatcher()函数的
# 第一个参数 normType 用于设置距离测量方法，默认为 cv2.NORM_L2，
# 如果使用ORB描述符(或其他基于二进制的描述符，例如 BRIEF或 BRISK)，则要使用的距离测量方法 cv2.NORM_HAMMING；
# 第二个参数 crossCheck (默认为 False) 可以设置为 True，以便在匹配过程中只返回两个集合中相互匹配的特征。

# 创建匹配器对象后，使用BFMatcher.match()方法匹配检测到的描述符：
bf_matches = bf_matcher.match(descriptors_1, descriptors_2)
# descriptors_1 和 descriptors_2 是计算得到的描述符；
# 返回值为两个图像中获得了最佳匹配，

# 我们可以按距离的升序对匹配项进行排序：
bf_matches = sorted(bf_matches, key=lambda x: x.distance)

# 最后，我们使用cv2.drawMatches()函数绘制匹配特征对，为了更好的可视化效果，仅显示前20个匹配项：
result = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, bf_matches[:20], None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)
# cv2.drawMatches()函数水平拼接两个图像，并绘制从第一个图像到第二个图像的线条以显示匹配特征对

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

plt.suptitle("ORB keypoint detector", fontsize=14, fontweight='bold')

show_img_with_matplotlib(result, "matches between the two images", 1)

plt.show()