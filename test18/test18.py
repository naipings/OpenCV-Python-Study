# 图像处理中的常用滤波器
# 还可以定义一些用于不同目的的通用核，例如：边缘检测、平滑、锐化或浮雕等，定义内核过后，可以使用cv2.filter2D()函数：

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069188

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test18/imgs/test01.jpeg')

kernel_identity = np.array([[0, 0, 0], 
                            [0, 1, 0], 
                            [0, 0, 0]])

# 边缘检测
kernel_edge_detection_1 = np.array([[1, 0, -1],
                                    [0, 0, 0],
                                    [-1, 0, 1]])

kernel_edge_detection_2 = np.array([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])

kernel_edge_detection_3 = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]])

# 锐化
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

kernel_unsharp_masking = -1 / 256 * np.array([[1, 4, 6, 4, 1],
                                              [4, 16, 24, 16, 4],
                                              [6, 24, -476, 24, 6],
                                              [4, 16, 24, 16, 4],
                                              [1, 4, 6, 4, 1]])

# 模糊
kernel_blur = 1 / 9 * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

gaussian_blur = 1 / 16 * np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]])

# 浮雕
kernel_emboss = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])

# 边缘检测
sobel_x_kernel = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])

sobel_y_kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

outline_kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

# 应用卷积核
original_image = cv2.filter2D(image, -1, kernel_identity)
edge_image_1 = cv2.filter2D(image, -1, kernel_edge_detection_1)
edge_image_2 = cv2.filter2D(image, -1, kernel_edge_detection_2)
edge_image_3 = cv2.filter2D(image, -1, kernel_edge_detection_3)
sharpen_image = cv2.filter2D(image, -1, kernel_sharpen)
unsharp_masking_image = cv2.filter2D(image, -1, kernel_unsharp_masking)
blur_image = cv2.filter2D(image, -1, kernel_blur)
gaussian_blur_image = cv2.filter2D(image, -1, gaussian_blur)
emboss_image = cv2.filter2D(image, -1, kernel_emboss)
sobel_x_image = cv2.filter2D(image, -1, sobel_x_kernel)
sobel_y_image = cv2.filter2D(image, -1, sobel_y_kernel)
outline_image = cv2.filter2D(image, -1, outline_kernel)

def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(3, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
    plt.suptitle("Sharpening images", fontsize=14, fontweight='bold')
show_with_matplotlib(original_image, 'identity kernel', 1)
show_with_matplotlib(edge_image_1, 'edge detection 1', 2)
show_with_matplotlib(edge_image_2, 'edge detection 1', 3)
show_with_matplotlib(edge_image_3, 'edge detection 1', 4)
show_with_matplotlib(sharpen_image, 'sharpen', 1+4)
show_with_matplotlib(unsharp_masking_image, 'unsharp masking', 2+4)
show_with_matplotlib(blur_image, 'blur image', 3+4)
show_with_matplotlib(gaussian_blur_image, 'gaussian blur image', 4+4)
show_with_matplotlib(emboss_image, 'emboss image', 1+4*2)
show_with_matplotlib(sobel_x_image, 'sobel x image', 2+4*2)
show_with_matplotlib(sobel_y_image, 'sobel y image', 3+4*2)
show_with_matplotlib(outline_image, 'outline image', 4+4*2)
plt.show()