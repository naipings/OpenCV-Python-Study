# 膨胀运算与腐蚀运算

# 二值图像的膨胀运算的主要作用是扩大前景对象的边界区域，
# 这意味着前景对象的区域会变大，而这些区域的孔会缩小：
# dilation = cv2.dilate(image, kernel, iterations=1)

# 腐蚀操作对二值图像的主要作用是逐渐侵蚀掉前景对象的边界区域，
# 这意味着前景对象的区域会变小，而这些区域内的空洞会变大：
# erosion = cv2.erode(image, kernel, iterations=1)

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image_names = ['test1.png', 'test2.png', 'test3.png']
path = '../opencvStudy/test20/imgs'

kernel_size_3_3 = (3, 3)
kernel_size_5_5 = (5, 5)

def load_all_test_images():
    test_morph_images = []
    for index_image, name_image in enumerate(image_names):
        image_path = os.path.join(path, name_image)
        test_morph_images.append(cv2.imread(image_path))
    return test_morph_images

def build_kernel(kernel_type, kernel_size):
    """创建执行形态学运算时要使用的核"""
    if kernel_type == cv2.MORPH_ELLIPSE:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_type == cv2.MORPH_CROSS:
        return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

def erode(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    erosion = cv2.erode(image, kernel, iterations=1)
    return erosion

def dilate(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    dilation = cv2.dilate(image, kernel, iterations=1)
    return dilation

test_images = load_all_test_images()

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

for index_image,image in enumerate(test_images):
    show_with_matplotlib(image, 'test img_{}'.format(index_image + 1), index_image * 3 + 1)
    img_1 = erode(image, cv2.MORPH_RECT, (3,3))
    show_with_matplotlib(img_1, 'erode_{}'.format(index_image + 1), index_image * 3 + 2)
    img_2 = dilate(image, cv2.MORPH_RECT, (3,3))
    show_with_matplotlib(img_2, 'dilate_{}'.format(index_image + 1), index_image * 3 + 3)
plt.show()