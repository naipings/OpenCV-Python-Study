# 利用特征匹配和单应性计算以查找对象
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 见test49-2.py
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
# 创建匹配器对象后，使用BFMatcher.match()方法匹配检测到的描述符：
bf_matches = bf_matcher.match(descriptors_1, descriptors_2)
# 我们可以按距离的升序对匹配项进行排序：
bf_matches = sorted(bf_matches, key=lambda x: x.distance)


# 然后，我们将利用上述特征匹配和单应性计算查找对象。
# 为了达到此目的，在完成特征匹配后，下一步需要使用 cv2.findHomography()函数在两幅图像中找到匹配关键点位置之间的透视变换。
# OpenCV提供了多种计算单应矩阵的方法，包括 RANSAC、最小中值(LMEDS)和PROSAC(RHO)。
# 我们以使用RANSAC方法为例：
# 提取匹配的关键点
pts_src = np.float32([keypoints_1[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
pts_dst = np.float32([keypoints_2[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
# 计算单应性矩阵
M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
# 其中，pts_src是匹配的关键点在源图像中的位置，
# pts_dst是匹配的关键点在查询图像中的位置。
# 第四个参数 ransacReprojThreshold设置最大重投影误差以将点对视为内点，如果重投影误差大于 5.0，则相应的点对被视为异常值。
# 此函数计算并返回由关键点位置定义的源平面和目标平面之间的透视变换矩阵 M。


# 最后，基于透视变换矩阵 M，计算查询图像中对象的四个角，用于绘制匹配的目标边界框。
# 为此，需要根据原始图像的形状计算其四个角，并使用 cv2.perspectiveTransform()函数将它们转换为目标角：
# 获取“查询”图像的角坐标
h, w = image_1.shape[:2]
pts_corners_src = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# 使用矩阵 M 和“查询”图像的角点执行透视变换，以获取“场景”图像中“检测”对象的角点：
pts_corners_dst = cv2.perspectiveTransform(pts_corners_src, M)
# 其中，pts_corners_src包含原图的四个角坐标，
# M为透视变换矩阵；
# pts_corners_dst输出包含查询图像中对象的四个角，

# 之后，我们可以使用cv2.polyline()函数来绘制检测对象的轮廓;
img_obj = cv2.polylines(image_2, [np.int32(pts_corners_dst)], True, (0, 255, 255), 10)

# 最后，使用cv2.drawMatches()函数绘制匹配特征点：
img_matching = cv2.drawMatches(image_1, keypoints_1, img_obj, keypoints_2, bf_matches, None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

plt.suptitle("Feature matching and homography computation for object recognition", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_obj, "detected object", 1)
show_img_with_matplotlib(img_matching, "feature matching", 2)

plt.show()