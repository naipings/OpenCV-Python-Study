# 简单的阈值处理技术
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 了解cv2.threshold()不同参数的工作原理后，我们将cv2.threshold()应用于真实图像，并使用不同的阈值：
# 加载图像
image = cv2.imread('../opencvStudy/test33/imgs/test01.jpeg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Thresholding introduction", fontsize=14, fontweight='bold')

# 绘制灰度图像
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img", 1)

# 使用不同的阈值调用 cv2.threshold() 并进行可视化
for i in range(8):
    ret, thresh = cv2.threshold(gray_image, 130 + i * 10, 255, cv2.THRESH_BINARY)
    show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = {}".format(130 + i * 10), i + 2)

plt.show()

# 如上图所示，阈值在使用cv2.threshold()对图像进行阈值处理时起着至关重要的作用。
# 假设图像处理算法用于识别图像中的对象，如果阈值较低，则阈值图像中缺少一些信息，而如果阈值较高，则有一些信息会被黑色像素遮挡。
# 因此，为整个图像找到一个全局最优阈值是相当困难的，特别是如果图像受到不同光照条件的影响，找到全局最优阈值几乎是不可能的。
# 这就是为什么我们需要应用其他自适应阈值算法来对图像进行阈值处理的原因。