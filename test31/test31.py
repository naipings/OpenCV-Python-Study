# 直方图的比较
# ————cv2.compareHist()用于计算两个直方图的匹配程度。
# 详细原理可见截屏，或者见网页。

# 该函数用法：
# cv2.compareHist(H1, H2, method)

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 为了对比不同的度量方法，我们首先加载图像并对其进行变换，
# 然后使用所有度量方法计算这些图像和测试图像之间的相似度。
# 加载图像
image = cv2.imread('../opencvStudy/test31/imgs/test01.jpeg')
# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

M = np.ones(gray_image.shape, dtype='uint8') * 30
# 所有像素值加上 30
added_image = cv2.add(gray_image, M)
# 所有像素值减去 30
subtracted_image = cv2.subtract(gray_image, M)
# 使用模糊滤镜
blurred_image = cv2.blur(gray_image, (10,10))

def load_all_test_images():
    images = []
    images.append(gray_image)
    images.append(added_image)
    images.append(subtracted_image)
    images.append(blurred_image)
    return images

test_images = load_all_test_images()

# （注：hist_color_img()相关知识见：test26）
def hist_color_img(img):
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr
# 计算直方图均衡前后的直方图
hists = hist_color_img(image)

# 使用四种不同的度量方法计算这些图像和测试图像之间的相似度：
for img in test_images:
    # 计算直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 直方图归一化
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    hists.append(hist)
# 使用　cv2.HISTCMP_CORREL　度量方法
gray_gray_1 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
gray_grayblurred_1 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
gray_addedgray_1 = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CORREL)
gray_subgray_1 = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CORREL)
# 使用　cv2.HISTCMP_CHISQR　度量方法
gray_gray_2 = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_CHISQR)
gray_grayblurred_2 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CHISQR)
gray_addedgray_2 = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CHISQR)
gray_subgray_2 = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CHISQR)
# 使用　cv2.HISTCMP_INTERSECT　度量方法
gray_gray_3 = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_INTERSECT)
gray_grayblurred_3 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_INTERSECT)
gray_addedgray_3 = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_INTERSECT)
gray_subgray_3 = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_INTERSECT)
# 使用　cv2.HISTCMP_BHATTACHARYYA　度量方法
gray_gray_4 = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_BHATTACHARYYA)
gray_grayblurred_4 = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_BHATTACHARYYA)
gray_addedgray_4 = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_BHATTACHARYYA)
gray_subgray_4 = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_BHATTACHARYYA)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(1, 4, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


# show_hist_with_matplotlib_gray(gray_gray_1, "grayscale histogram", 1, 'm')
# show_hist_with_matplotlib_gray(gray_grayblurred_1, "grayscale histogram", 2, 'm')
# show_hist_with_matplotlib_gray(gray_addedgray_1, "grayscale histogram", 3, 'm')
# show_hist_with_matplotlib_gray(gray_subgray_1, "grayscale histogram", 4, 'm')

print(gray_gray_1)

# show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
# show_img_with_matplotlib(cv2.cvtColor(gray_gray_1, cv2.COLOR_GRAY2BGR), "grayscale equalized", 2)
# show_img_with_matplotlib(cv2.cvtColor(gray_grayblurred_1, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 3)
# show_img_with_matplotlib(cv2.cvtColor(gray_addedgray_1, cv2.COLOR_GRAY2BGR), "gray", 4)
# show_img_with_matplotlib(cv2.cvtColor(gray_subgray_1, cv2.COLOR_GRAY2BGR), "grayscale equalized", 5)

# show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 1+5)
# show_img_with_matplotlib(cv2.cvtColor(gray_gray_2, cv2.COLOR_GRAY2BGR), "gray", 2+5)
# show_img_with_matplotlib(cv2.cvtColor(gray_grayblurred_2, cv2.COLOR_GRAY2BGR), "grayscale equalized", 3+5)
# show_img_with_matplotlib(cv2.cvtColor(gray_addedgray_2, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 4+5)
# show_img_with_matplotlib(cv2.cvtColor(gray_subgray_2, cv2.COLOR_GRAY2BGR), "gray", 5+5)

# show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "grayscale equalized", 1+5*2)
# show_img_with_matplotlib(cv2.cvtColor(gray_gray_3, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 2+5*2)
# show_img_with_matplotlib(cv2.cvtColor(gray_grayblurred_3, cv2.COLOR_GRAY2BGR), "gray", 3+5*2)
# show_img_with_matplotlib(cv2.cvtColor(gray_addedgray_3, cv2.COLOR_GRAY2BGR), "grayscale equalized", 4+5*2)
# show_img_with_matplotlib(cv2.cvtColor(gray_subgray_3, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 5+5*2)

# show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1+5*3)
# show_img_with_matplotlib(cv2.cvtColor(gray_gray_4, cv2.COLOR_GRAY2BGR), "grayscale equalized", 2+5*3)
# show_img_with_matplotlib(cv2.cvtColor(gray_grayblurred_4, cv2.COLOR_GRAY2BGR), "grayscale CLAHE", 3+5*3)
# show_img_with_matplotlib(cv2.cvtColor(gray_addedgray_4, cv2.COLOR_GRAY2BGR), "gray", 4+5*3)
# show_img_with_matplotlib(cv2.cvtColor(gray_subgray_4, cv2.COLOR_GRAY2BGR), "grayscale equalized", 5+5*3)


# plt.show()

