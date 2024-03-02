# 显示色彩空间

# 参见：https://blog.csdn.net/LOVEmy134611/article/details/120069317

# 在以下示例中，图像被加载到BGR色彩空间并转换为上述色彩空间。
# 在此脚本中，关键函数是cv2.cvColor()，它可以将一种色彩空间的输入图像转换成另一种色彩空间。
import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

image = cv2.imread('../opencvStudy/test20/imgs/test01.jpeg')

def show_with_matplotlib(color_img, title, pos):
    # 一般显示用这个：
    # img_RGB = color_img[:, :]
    
    # 以灰度图像显示用这个：
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(4, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
plt.figure(figsize=(12, 5))
plt.suptitle("Color spaces in OpenCV", fontsize=14, fontweight='bold')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(bgr_b, bgr_g, bgr_r) = cv2.split(image)

# 转换为 HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
(hsv_h, hsv_s, hsv_v) = cv2.split(hsv_image)

# 转换为 HLS
hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
(hls_h, hls_l, hls_s) = cv2.split(hls_image)

# 转换为 YCrCb
ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
(ycrcb_y, ycrcb_cr, ycrcb_cb) = cv2.split(ycrcb_image)

show_with_matplotlib(image, "BGR - image", 2)

# Show gray image:
show_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray image", 1)

# 显示 RGB 分量通道
show_with_matplotlib(cv2.cvtColor(bgr_b, cv2.COLOR_GRAY2BGR), "BGR - B comp", 2 + 6)
show_with_matplotlib(cv2.cvtColor(bgr_g, cv2.COLOR_GRAY2BGR), "BGR - G comp", 2 + 6 * 2)
show_with_matplotlib(cv2.cvtColor(bgr_r, cv2.COLOR_GRAY2BGR), "BGR - R comp", 2 + 6 * 3)

# 展示其他色彩空间分量通道
# 显示 HSV 及其分量通道
# show_with_matplotlib(hsv_image, "HSV - image", 3)
# show_with_matplotlib(hsv_h, "HSV - h comp", 3 + 6)
# show_with_matplotlib(hsv_s, "HSV - s comp", 3 + 6 * 2)
# show_with_matplotlib(hsv_v, "HSV - v comp", 3 + 6 * 3)
# 以灰度图像显示：
show_with_matplotlib(cv2.cvtColor(hsv_h, cv2.COLOR_GRAY2BGR), "HSV - h comp", 3 + 6)
show_with_matplotlib(cv2.cvtColor(hsv_s, cv2.COLOR_GRAY2BGR), "HSV - s comp", 3 + 6 * 2)
show_with_matplotlib(cv2.cvtColor(hsv_v, cv2.COLOR_GRAY2BGR), "HSV - v comp", 3 + 6 * 3)

# 显示 HLS 及其分量通道
# show_with_matplotlib(hls_image, "HLS - image", 4)
# show_with_matplotlib(hls_h, "HLS - h comp", 4 + 6)
# show_with_matplotlib(hls_l, "HLS - l comp", 4 + 6 * 2)
# show_with_matplotlib(hls_s, "HLS - s comp", 4 + 6 * 3)
# 以灰度图像显示：
show_with_matplotlib(cv2.cvtColor(hls_h, cv2.COLOR_GRAY2BGR), "HLS - h comp", 4 + 6)
show_with_matplotlib(cv2.cvtColor(hls_l, cv2.COLOR_GRAY2BGR), "HLS - l comp", 4 + 6 * 2)
show_with_matplotlib(cv2.cvtColor(hls_s, cv2.COLOR_GRAY2BGR), "HLS - s comp", 4 + 6 * 3)

# 显示 YCrCb 及其分量通道
# show_with_matplotlib(ycrcb_image, "YCrCb - image", 5)
# show_with_matplotlib(ycrcb_y, "YCrCb - h comp", 5 + 6)
# show_with_matplotlib(ycrcb_cr, "YCrCb - s comp", 5 + 6 * 2)
# show_with_matplotlib(ycrcb_cb, "YCrCb - v comp", 5 + 6 * 3)
# 以灰度图像显示：
show_with_matplotlib(cv2.cvtColor(ycrcb_y, cv2.COLOR_GRAY2BGR), "YCrCb - h comp", 5 + 6)
show_with_matplotlib(cv2.cvtColor(ycrcb_cr, cv2.COLOR_GRAY2BGR), "YCrCb - s comp", 5 + 6 * 2)
show_with_matplotlib(cv2.cvtColor(ycrcb_cb, cv2.COLOR_GRAY2BGR), "YCrCb - v comp", 5 + 6 * 3)

plt.show()


# 注：在进行色彩空间转换时，应明确指定通道的顺序（BGR或RGB）：
# 将图像加载到 BGR 色彩空间中
# image = cv2.imread('color_spaces.png')
# 将其转换为 HSV 色彩空间
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# ————可以看到，此处使用了cv2.COLOR_BGR2HSV而不是cv2.COLOR_RGB2HSV