# Hu不变矩
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 接下来，为了对比Hu不变矩，我们使用三个图像。
# 第一个是原始图像，第二个将原始图像旋转180度，第三个将原始图像水平翻转，计算上述图像的 Hu不变矩。
# 程序的第一步是使用 cv2.imread()加载图像，并通过使用 cv2.cvtColor()将它们转换为灰度图像。
# 第二步是应用 cv2.threshold()获取二进制图像。
# 最后，使用 cv2.HuMoments()计算Hu不变矩：

# 加载图像并进行转换
image_1 = cv2.imread("../opencvStudy/test43/imgs/test03.jpg")
gray_image = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image_2 = cv2.flip(image_1, -1) 
image_3 = cv2.flip(image_1, 1)

images = [image_1, image_2, image_3]
des = ['original', 'rotation', 'reflection']

# 可视化函数
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

for i in range(len(images)):
    image = images[i]
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 图像二值化
    ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    # 计算 Hu 不变矩
    HuM_1 = cv2.HuMoments(cv2.moments(thresh, True)).flatten()
    # 打印 Hu 不变矩
    print("Hu moments ({}): '{}'".format(des[i], HuM_1))
    # 可视化
    show_img_with_matplotlib(image, "original", 1+i )

plt.show()

# 查看计算的Hu不变矩结果：
# Hu moments (original): '[ 2.14889509e-01  1.16653928e-02  7.70456336e-06  8.92045133e-06  -3.09852503e-11  7.90880440e-07 -6.71485049e-11]'
# Hu moments (rotation): '[ 2.14889509e-01  1.16653928e-02  7.70456336e-06  8.92045133e-06 -3.09852503e-11  7.90880440e-07 -6.71485049e-11]'
# Hu moments (reflection): '[ 2.14889509e-01  1.16653928e-02  7.70456336e-06  8.92045133e-06 -3.09852503e-11  7.90880440e-07  6.71485049e-11]'

# 可以看到，除了第七个矩外，计算的Hu不变矩在这三种情况下是相同的。