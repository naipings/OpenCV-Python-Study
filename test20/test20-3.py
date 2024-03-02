# 形态梯度运算

# 形态梯度运算定义为：输入图像的膨胀和腐蚀之间的差异：
# morph_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

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

def morphological_gradient(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    morph_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return morph_gradient
    
test_images = load_all_test_images()

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

for index_image,image in enumerate(test_images):
    print(index_image)
    show_with_matplotlib(image, 'test img_{}'.format(index_image + 1), index_image + 1)
    img = morphological_gradient(image, cv2.MORPH_RECT, (3,3))
    show_with_matplotlib(img, 'gradient_{}'.format(index_image + 1), index_image + 4)
plt.show()

