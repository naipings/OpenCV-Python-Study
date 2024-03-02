# 带有蒙版的灰度图像
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 如果需要应用蒙版，需要首先创建一个蒙版：
# 加载并修改图像
image = cv2.imread('../opencvStudy/test25/imgs/test2.jpg')
height, width = image.shape[:2]

# 添加了一些具有 0 和 255 灰度强度的黑色和白色小圆圈
for i in range(0, width, 20):
    cv2.circle(image, (i, 390), 10, (0, 0, 0), -1)
    cv2.circle(image, (i, 3500), 10, (255, 255, 255), -1)
    # 将图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 计算原灰度图像直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# 创建蒙版
mask = np.zeros(gray_image.shape[:2], np.uint8)
mask[410:3480, 100:2100] = 255 #等号左边的值：分别对应：高范围、宽范围

# 蒙版由与加载图像尺寸相同的黑色图像组成，白色图像对应于要计算直方图的区域。
# 然后使用所创建的蒙版来计算直方图，调用cv2.calcHist()函数并传递创建的蒙版：
hist_mask = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])
masked_img = cv2.bitwise_and(gray_image, gray_image, mask=mask)
# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(2, 2, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(14, 10))
plt.suptitle("Grayscale histograms introduction", fontsize=14, fontweight='bold')

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 2, 'm')
show_img_with_matplotlib(cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR), "masked gray image", 3)
show_hist_with_matplotlib_gray(hist_mask, "grayscale masked histogram", 4, 'm')

plt.show()

# 如效果图所示，我们对图像进行了修改，我们在其中分别添加了一些黑色和白色的小圆圈，这导致直方图在bins=0和255中有较大的值，如第一个直方图所示。
# 但是，添加的这些修改不会出现在蒙版图像直方图中，因为应用了蒙版，因此在计算直方图时没有将它们考虑在内。