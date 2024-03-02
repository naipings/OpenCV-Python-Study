# 自定义色彩映射

# 第一种方法是：定义一个色彩映射，将0到255个灰度值映射到256种颜色。
# 这可以通过创建大小为256*1的8位色彩图像来完成，以便存储所有车间内的颜色。
# 之后，可以用以下方法 通过查找表 将图像的灰度强度 映射到定义的颜色：

# 1.使用cv2.LUT()函数
# 2.使用cv2.applyColorMap()函数
#  ————需要注意的是，在创建大小为256*1的8位彩色图像用于存储图像时，如果打算使用cv2.LUT()，则应按如下方式创建图像：
# lut = np.zeros((256, 3), dtype=np.uint8)
# 如果打算使用cv2.applyColorMap()，则应使用：
# lut = np.zeros((256, 1, 3), dtype=np.uint8)

# 完整的代码如下：
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def apply_rand_custom_colormap_values(im_gray):
    lut = np.random.randint(255, size=(256, 1, 3), dtype=np.uint8)

    im_color = cv2.applyColorMap(im_gray, lut)
    return im_color

def apply_rand_custom_colormap_values2(im_gray):
    # 创建随机 LUT
    lut = np.random.randint(255, size=(256, 3), dtype=np.uint8)

    # 使用 cv2.LUT() 应用自定义色彩映射
    s0, s1 = im_gray.shape
    im_color = np.empty(shape=(s0, s1, 3), dtype=np.uint8)
    for i in range(3):
        im_color[..., i] = cv2.LUT(im_gray, lut[:, i])
    return im_color

def show_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 5, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')


# 读取图像并转化为灰度图像
gray_img = cv2.cvtColor(cv2.imread('../opencvStudy/test21/imgs/test2.jpg'), cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 2))
plt.suptitle("Custom colormaps providing all values", fontsize=14, fontweight='bold')

show_with_matplotlib(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), "gray", 1)

# 应用色彩映射
custom_rand_1 = apply_rand_custom_colormap_values(gray_img)
custom_rand_2 = apply_rand_custom_colormap_values2(gray_img)
# 可以自行创建固定值色彩映射并应用，与随机创建类似，在此不再赘述
# custom_values_1 = apply_custom_colormap_values(gray_img)
# custom_values_2 = apply_custom_colormap_values2(gray_img)

# 可视化
show_with_matplotlib(custom_rand_1, "cv2.applyColorMap()", 2)
show_with_matplotlib(custom_rand_2, "cv2.LUT()", 3)

plt.show()
