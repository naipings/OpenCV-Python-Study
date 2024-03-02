# 顶帽运算与低帽（黑帽运算）

# 顶帽运算被定义为输入图像和图像开运算之间的差：
# top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# 低帽（黑帽）操作被定义为输入图像和输入图像闭运算的差：
# black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image_names = ['test1.png', 'test2.png', 'test3.png']
path = '../opencvStudy/test20/imgs'

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

# build_kernel() 和 show_with_matplotlib() 函数与test20-1中相同
def top_hat(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    top = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return top

def black_hat(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    black = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return black

def closing(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    clos = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return clos

def opening(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    ope = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return ope

def opening_and_closing(image, kernel_type, kernel_size):
    opening_img = opening(image, kernel_type, kernel_size)
    closing_img = closing(opening_img, kernel_type, kernel_size)
    return closing_img
    
test_images = load_all_test_images()

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

for index_image,image in enumerate(test_images):
    show_with_matplotlib(image, 'test img_{}'.format(index_image + 1), index_image * 3 + 1)
    img_1 = top_hat(image, cv2.MORPH_RECT, (3,3))
    show_with_matplotlib(img_1, 'top_hat_{}'.format(index_image + 1), index_image * 3 + 2)
    img_2 = black_hat(image, cv2.MORPH_RECT, (3,3))
    show_with_matplotlib(img_2, 'black_hat_{}'.format(index_image + 1), index_image * 3 + 3)
plt.show()

