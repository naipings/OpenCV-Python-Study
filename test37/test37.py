# 对彩色图像进行阈值处理
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# cv2.threshold()函数也可应用于多通道图像，在此情况下，cv2.threshold()函数在BGR图像的每个通道中应用阈值操作：
image = cv2.imread('../opencvStudy/test36/imgs/test2.jpg')
ret1, thresh1 = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

# 这与在每个通道中应用cv2.threshold()函数并合并阈值通道结果相同：
(b, g, r) = cv2.split(image)
ret2, thresh2 = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(r, 150, 255, cv2.THRESH_BINARY)
bgr_thresh = cv2.merge((thresh2, thresh3, thresh4))

# 为了进行对比，我们同时使用其他阈值类型对彩色图像进行阈值处理：
ret5, thresh5 = cv2.threshold(image, 120, 255, cv2.THRESH_TOZERO)

# Apply cv2.threshold():
(b, g, r) = cv2.split(image)
ret6, thresh6 = cv2.threshold(b, 120, 255, cv2.THRESH_TOZERO)
ret7, thresh7 = cv2.threshold(g, 120, 255, cv2.THRESH_TOZERO)
ret8, thresh8 = cv2.threshold(r, 120, 255, cv2.THRESH_TOZERO)
bgr_thresh_2 = cv2.merge((thresh2, thresh3, thresh4))

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
plt.suptitle("Triangle's binarization algorithm applying a Gaussian filter", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image, "image", 1)

show_img_with_matplotlib(thresh1, "threshold(120) BGR image", 2)
show_img_with_matplotlib(bgr_thresh, "threshold(120) each channel and merge", 3)
show_img_with_matplotlib(thresh5, "threshold(cv2.THRESH_TOZERO 120) BGR image", 5)
show_img_with_matplotlib(bgr_thresh_2, "threshold(cv2.THRESH_TOZERO 120) each channel and merge", 6)

plt.show()