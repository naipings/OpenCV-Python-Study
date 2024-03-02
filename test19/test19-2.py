# 图像加减法与图像混合
# ————cv2.add()  cv2.subtract() 

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test19/imgs/test01.jpeg')

# 如果：对图像的所有像素添加60，首先要构建图像以添加到原始图像：
M = np.ones(image.shape, dtype="uint8") * 60

# 然后，使用以下代码执行加法或减法：
added_image = cv2.add(image, M)
subtracted_image = cv2.subtract(image, M)


# 也可以创建一个标量并将其添加到原始图像中。
# 例如，要给图像的所有像素加上110，首先构建标量：
scalar = np.ones((1, 3), dtype="float") * 110

# 然后，使用以下代码执行加法或减法：
added_image_2 = cv2.add(image, scalar)
subtracted_image_2 = cv2.subtract(image, scalar)


def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    # plt.axis('off')
show_with_matplotlib(image, 'image', 1)
show_with_matplotlib(added_image, 'added 60 (image + image)', 2)
show_with_matplotlib(subtracted_image, 'subtracted 60 (image - images)', 3)
show_with_matplotlib(added_image_2, 'added 110 (image + scalar)', 2+3)
show_with_matplotlib(subtracted_image_2, 'subtracted 110 (image - scalar)', 3+3)
plt.show()
