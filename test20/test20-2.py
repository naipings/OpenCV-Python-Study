# 开运算与闭运算

# 开运算先执行腐蚀，然后使用相同的结构元素（或核）。
# 通过这种方式，可以应用腐蚀来消除一小组不需要的像素（例如，椒盐噪声）。

# 腐蚀会不分青红皂白地影响图像的所有区域。
# 通过在腐蚀后执行扩张操作，可以减少腐蚀过度的一些影响：
# opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# 闭运算同样也可以从腐蚀和膨胀操作中推导出来，该操作先执行膨胀，然后执行腐蚀。
# 膨胀操作通常用于填充图像中的小孔。然而，膨胀操作也会使一小群噪声像素变大。
# 通过在膨胀后对图像应用腐蚀操作，将减少膨胀带来的这种影响：
# closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

# 接下来，实际使用开运算与闭运算：
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
def closing(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    clos = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return clos

def opening(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    ope = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return ope

test_images = load_all_test_images()

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

for index_image,image in enumerate(test_images):
    show_with_matplotlib(image, 'test img_{}'.format(index_image + 1), index_image * 3 + 1)
    img_1 = closing(image, cv2.MORPH_RECT, (3,3))
    show_with_matplotlib(img_1, 'closing_{}'.format(index_image + 1), index_image * 3 + 2)
    img_2 = opening(image, cv2.MORPH_RECT, (3,3))
    show_with_matplotlib(img_2, 'opening_{}'.format(index_image + 1), index_image * 3 + 3)
plt.show()
