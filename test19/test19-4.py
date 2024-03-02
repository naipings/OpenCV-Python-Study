# 按位运算
# 按位运算包括：AND、OR、NOT和XOR

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 为了演示按位运算，我们首先创建一些图像：
img_1 = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(img_1, (10, 10), (110, 110), (255, 255, 255), -1)
cv2.circle(img_1, (200, 200), 50, (255, 255, 255), -1)

img_2 = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(img_2, (50, 50), (150, 150), (255, 255, 255), -1)
cv2.circle(img_2, (225, 200), 50, (255, 255, 255), -1)

image = cv2.imread('../opencvStudy/test19/imgs/test01.jpeg')
image = cv2.resize(image,(300, 300))

img_3 = np.zeros((300, 300), dtype="uint8")
cv2.circle(img_3, (150, 150), 150, (255, 255, 255), -1)

# 然后对所创建的图像进行按位运算：
# OR
bitwise_or = cv2.bitwise_or(img_1, img_2)
# AND
bitwise_and = cv2.bitwise_and(img_1, img_2)
# XOR
bitwise_xor = cv2.bitwise_xor(img_1, img_2)
# NOT
bitwise_not_1 = cv2.bitwise_not(img_1)
bitwise_not_2 = cv2.bitwise_not(img_2)
# AND with mask
bitwise_and_example = cv2.bitwise_and(image, image, mask=img_3)

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :]
    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
plt.suptitle("Bitwise operations(AND, OR, XOR, NOT)", fontsize=14, fontweight='bold')
show_with_matplotlib(img_1, "image 1", 1)
show_with_matplotlib(img_2, "image 2", 2)
show_with_matplotlib(bitwise_or, "image 1 OR image 2", 3)
show_with_matplotlib(bitwise_and, "image 1 AND image 2", 1+3)
show_with_matplotlib(bitwise_xor, "image 1 XOR image 2", 2+3)
show_with_matplotlib(bitwise_not_1, "NOT(image 1)", 3+3)
show_with_matplotlib(bitwise_not_2, "NOT(image 2)", 1+3*2)
show_with_matplotlib(img_3, "image 3", 2+3*2)
show_with_matplotlib(bitwise_and_example, "image 3 AND a loaded image", 3+3*2)
# Show the Figure:
plt.show()