# 自定义色彩映射
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 第二种方法是：仅提供一些关键颜色，然后对这些值进行插值，以获得构建查找表u所需的所有颜色。
# 编写build_lut()函数根据这些关键颜色构建查找表：基于5个预先定义的色点，调用np.linspace()在预定义的每个色点区间内计算均匀间隔的颜色：
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


# 然后应用自定义颜色映射：
def show_with_matplotlib(color_img, title, pos):
    # 通道转换
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

def apply_color_map_1(gray, cmap):

    lut = build_lut(cmap)
    s0, s1 = gray.shape
    out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)

    for i in range(3):
        out[..., i] = cv2.LUT(gray, lut[:, i])
    return out

def apply_color_map_2(gray, cmap):

    lut = build_lut(cmap)
    lut_reshape = np.reshape(lut, (256, 1, 3))
    im_color = cv2.applyColorMap(gray, lut_reshape)
    return im_color

# 读取图像并转化为灰度图像
gray_img = cv2.cvtColor(cv2.imread('../opencvStudy/test22/imgs/test2.jpg'), cv2.COLOR_BGR2GRAY)

# 应用色彩映射
custom_1 = apply_color_map_1(gray_img, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))

custom_2 = apply_color_map_1(gray_img, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))

custom_3 = apply_color_map_2(gray_img, ((0, (255, 0, 255)), (0.25, (255, 0, 180)), (0.5, (255, 0, 120)),
                                        (0.75, (255, 0, 60)), (1.0, (255, 0, 0))))

custom_4 = apply_color_map_2(gray_img, ((0, (0, 255, 128)), (0.25, (128, 184, 64)), (0.5, (255, 128, 0)),
                                        (0.75, (64, 128, 224)), (1.0, (0, 128, 255))))

# 可视化
show_with_matplotlib(custom_1, "custom 1 using cv2.LUT()", 2)
show_with_matplotlib(custom_2, "custom 2 using cv2.LUT()", 3)
show_with_matplotlib(custom_3, "custom 3 using cv2.applyColorMap()", 5)
show_with_matplotlib(custom_4, "custom 4 using using cv2.applyColorMap()", 6)

plt.show()

# 在上图中，可以看到将两个自定义色彩映射应用于灰度图像的效果。