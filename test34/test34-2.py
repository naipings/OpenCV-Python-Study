# 自适应阈值算法
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 根据34.py的效果图，我们可以看到应用具有不同参数的cv2.adaptiveThreshold()处理后的输出，自适应阈值可以提供更好的阈值图像。
# 但是，图像中也会出现了很多噪点。为了减少噪点，可以应用一些平滑操作，例如双边滤波，因为它会在去除噪声的同时保持锐利边缘。
# 因此，我们在对图像进行阈值处理之前首先应用 OpenCV 提供的双边滤波函数 cv2.bilateralFilter()：

# 加载图像
image = cv2.imread('../opencvStudy/test34/imgs/test01.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 双边滤波
gray_image = cv2.bilateralFilter(gray_image, 15, 25, 25)
# 应用自适应阈值处理
thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Adaptive thresholding applying a bilateral filter(noise removal while edges sharp)", fontsize=14, fontweight='bold')

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=11, C=2", 2)
# 其他图像可视化方法类似，不再赘述
show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=31, C=3", 3)
show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=11, C=2", 5)
show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=31, C=3", 6)

plt.show()

# 从结果图可以看到，应用平滑滤波器后，噪声数量大幅减少。