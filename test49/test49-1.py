# 特征检测
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 一个特征可以被描述为图像中的一小块区域，它对图像的缩放、旋转和照明尽可能保持不变。
# 因此，可以从同一场景不同视角的不同图像中检测到相同的特征。
# 综上，一个好的特征应具备以下特性：
#     1.可重复(可以从同一物体的不同图像中提取相同的特征)
#     2.可区分(不同结构的图像具有不同的特征)

# OpenCV提供了许多算法来检测图像的特征，包括：
# 哈里斯角检测 (Harris Corner Detection)、
# Shi-Tomasi角检测 (Shi-Tomasi Corner Detection)、
# SIFT (Scale Invariant Feature Transform)、
# SURF (Speeded-Up Robust Features)、
# FAST (Features from Accelerated Segment Test)、
# BRIEF (Binary Robust Independent Elementary Features) 
# ORB (Oriented FAST and Rotated BRIEF) 等。

# 以ORB算法为例，对图像进行特征检测和描述。ORB可以看作是FAST关键点检测器和BRIE描述符的组合，并进行了关键改进以提高性能。
# 首先是检测关键点，ORB使用修改后的FAST-9算法(radius = 9 ，并存储检测到的关键点的方向) 检测关键点(keypoints, 默认为 500个)。
# 一旦检测到关键点，下一步就是计算描述符以获得与每个检测到的关键点相关联的信息。
# ORB使用改进的 BRIEF-32 描述符来获取每个检测到的关键点的描述符。
# 检测到的关键点的描述符结构如下：
# [ 43 106  98  30 127 147 250  72  95  68 244 175  40 200 247 164 254 168 146 197 198 191  46 255  22  94 129 171  95  14 122 207]

# 根据上面讲解，第一步是创建ORB检测器：
orb = cv2.ORB_create()

# 然后是检测图像中的关键点：
# 加载图像
image = cv2.imread('../opencvStudy/test49/imgs/test16.png')

# 检测图像中的关键点
keypoints = orb.detect(image, None)

# 检测到关键点后，下一步就是计算检测到的关键点的描述符：
keypoints, descriptors = orb.compute(image, keypoints)
# 请注意，也可以执行通过orb.detectAndCompute(image, None)函数同时检测关键点并计算检测到的关键点的描述符。

# 最后，使用cv2.drawKeypoints()函数绘制检测到的关键点：
image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255), flags=0)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

plt.suptitle("ORB keypoint detector", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(image_keypoints, "detected keypoints", 2)

plt.show()