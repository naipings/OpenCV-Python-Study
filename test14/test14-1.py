import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 拆分与合并通道
# 使用cv2.split()将多通道图像拆分为多个单通道图像
# 使用cv2.merge()将多个单通道图像合并成一个多通道图像

# 通道拆分（使用cv2.split()函数，从加载的BGR图像中获取三个通道）
image = cv2.imread('../opencvStudy/test14/imgs/test01.jpeg')
(b, g, r) = cv2.split(image)

# 通道合并（使用cv2.merge()函数，将三个通道合并构建BGR图像）
image_merge = cv2.merge((b, g, r))

# 需要注意的是，cv2.split()是一个耗时的操作，使用应该只在绝对必要的时候使用它。
# 作为替代，可以使用NumPy切片语法来处理特定通道。例如，如果要获得图像的蓝色通道：
b = image[:, :, 0]

# 此外，可以消除多通道图像的某些通道（通过将通道值设置为0），得到的图像具有相同数量的通道，
# 但相应通道中的值为0；例如，如果要消除BGR图像的蓝色通道：
image_without_blue = image.copy()
image_without_blue[:, :, 0] = 0

# 消除其他通道的方法与上述代码原理相同：（针对BGR图像）
# 红蓝通道（即：消除绿色通道）
image_without_green = image.copy()
image_without_green[:,:,1] = 0
# 蓝绿通道（即：消除红色通道）
image_without_red = image.copy()
image_without_red[:,:,2] = 0

# 然后将得到的图像的通道拆分：
(b_1, g_1, r_1) = cv2.split(image_without_blue)
(b_2, g_2, r_2) = cv2.split(image_without_green)
(b_3, g_3, r_3) = cv2.split(image_without_red)

# 显示拆分后的通道：
def show_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:,:,::-1]

    ax = plt.subplot(3, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title,fontsize=8)
    plt.axis('off')
    

plt.figure(figsize=(13,5))
plt.suptitle('splitting and merging channels in OpenCV', fontsize=12, fontweight='bold')

show_with_matplotlib(image, "BGR - image", 1)
show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR - (B)", 2)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR - (G)", 2 + 6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR - (R)", 2 + 6 * 2)
show_with_matplotlib(image_merge, "BGR - image (merge)", 1 + 6)
show_with_matplotlib(image_without_blue, "BGR without B", 3)
show_with_matplotlib(image_without_green, "BGR without G", 3 + 6)
show_with_matplotlib(image_without_red, "BGR without R", 3 + 6 * 2)
show_with_matplotlib(cv2.cvtColor(b_1, cv2.COLOR_GRAY2BGR), "BGR without B (B)", 4)
show_with_matplotlib(cv2.cvtColor(g_1, cv2.COLOR_GRAY2BGR), "BGR without B (G)", 4 + 6)
show_with_matplotlib(cv2.cvtColor(r_1, cv2.COLOR_GRAY2BGR), "BGR without B (R)", 4 + 6 * 2)
# 显示其他拆分通道的方法完全相同，只需修改通道名和子图位置
# ...
# 本人将剩下的图像显示：
show_with_matplotlib(cv2.cvtColor(b_2, cv2.COLOR_GRAY2BGR), "BGR without G (B)", 5)
show_with_matplotlib(cv2.cvtColor(g_2, cv2.COLOR_GRAY2BGR), "BGR without G (G)", 5 + 6)
show_with_matplotlib(cv2.cvtColor(r_2, cv2.COLOR_GRAY2BGR), "BGR without G (R)", 5 + 6 * 2)

show_with_matplotlib(cv2.cvtColor(b_3, cv2.COLOR_GRAY2BGR), "BGR without R (B)", 6)
show_with_matplotlib(cv2.cvtColor(g_3, cv2.COLOR_GRAY2BGR), "BGR without R (G)", 6 + 6)
show_with_matplotlib(cv2.cvtColor(r_3, cv2.COLOR_GRAY2BGR), "BGR without R (R)", 6 + 6 * 2)

plt.show()

# 可能有些慢，运行代码后需稍等一下