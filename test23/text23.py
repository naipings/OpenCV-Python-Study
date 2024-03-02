# 显示自定义色彩映射图例

# 最后，我们也可以在显示自定义颜色映射时提供图例。
# 为了构建色彩映射图例，编写build_lut_image()：
# def build_lut_image(cmap, height):
#     lut = build_lut(cmap)
#     image = np.repeat(lut[np.newaxis, ...], height, axis=0)

#     return image
# 其首先调用build_lut()函数以获取查找表。然后调用np.repeat()以多次复制此查找表（height次）。
# 这是由于查找表的形状是（256, 3），而输出图像的形状为（height, 256, 3），为了增加新维度，我们还需要将np.repeat()与np.newaxis()一起使用：
# image = np.repeat(lut[np.newaxis, ...], height, axis=0)

# 完整的代码如下：
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# build_lut()函数功能介绍可以见：test22-2-2.py
def build_lut(cmap):
    lut = np.empty(shape=(256, 3), dtype=np.uint8)

    max = 256
    # 构建查找表
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * max)
        for i in range(3):                                                                     
            lut[lastval:val, i] = np.linspace(lastcol[i], col[i], val - lastval)
        lastcol = col
        lastval = val

    return lut

# 构建色彩映射图例
def build_lut_image(cmap, height):
    lut = build_lut(cmap)
    image = np.repeat(lut[np.newaxis, ...], height, axis=0)

    return image

def apply_color_map_1(gray, cmap):

    lut = build_lut(cmap)
    s0, s1 = gray.shape
    out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)

    for i in range(3):
        out[..., i] = cv2.LUT(gray, lut[:, i])
    return out

def show_with_matplotlib(color_img, title, pos):
    # 通道转换
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 读取图像并转化为灰度图像
gray_img = cv2.cvtColor(cv2.imread('../opencvStudy/test23/imgs/test2.jpg'), cv2.COLOR_BGR2GRAY)

# 应用色彩映射图例
custom_1 = build_lut_image(((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))), 20)

custom_2 = build_lut_image(((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))), 20)

custom_3 = apply_color_map_1(gray_img, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))

custom_4 = apply_color_map_1(gray_img, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))

# 可视化
show_with_matplotlib(custom_1, "", 1)
show_with_matplotlib(custom_2, "", 2)
show_with_matplotlib(custom_3, "", 3)
show_with_matplotlib(custom_4, "", 4)

plt.show()

# 在上图中，可以看到将两个自定义颜色映射应用于灰度图像并显示每个颜色映射的图例的效果。
