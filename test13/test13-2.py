# OpenCV文本字体
# 参见：https://blog.csdn.net/LOVEmy134611/article/details/119712265

# OpenCV中所有可用的字体如下：
# FONT_HERSHEY_SIMPLEX = 0
# FONT_HERSHEY_PLAIN = 1
# FONT_HERSHEY_DUPLEX = 2
# FONT_HERSHEY_COMPLEX = 3
# FONT_HERSHEY_TRIPLEX = 4
# FONT_HERSHEY_COMPLEX_SMALL = 5
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6
# FONT_HERSHEY_SCRIPT_COMPLEX = 7

#由于所有这些字体都在（0-7）范围内，我们可以迭代并调用cv2.putText()函数，改变color、fontFace好人org参数，使用所有可用字体进行绘制：
import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def show_with_matplotlib(img, title):
    # 将 BGR 图像转换为 RGB
    img_RGB = img[:, :, ::-1]
    # 使用 Matplotlib 显示图形
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()

# 定义颜色字典:
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
        'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
        'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
        'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# 给字体和颜色编号：
fonts = {0: "FONT HERSHEY SIMPLEX", 1: "FONT HERSHEY PLAIN", 2: "FONT HERSHEY DUPLEX", 3: "FONT HERSHEY COMPLEX",
        4: "FONT HERSHEY TRIPLEX", 5: "FONT HERSHEY COMPLEX SMALL ", 6: "FONT HERSHEY SCRIPT SIMPLEX",
        7: "FONT HERSHEY SCRIPT COMPLEX"}
index_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow', 4: 'magenta', 5: 'cyan', 6: 'black', 7: 'dark_gray'}

# 创建画布
image = np.zeros((650, 650, 3), dtype="uint8")

# 修改画布背景颜色，这里设置为灰色
image[:] = colors['gray']

position = (10, 30)
for i in range(0, 8):
	cv2.putText(image, fonts[i], position, i, 1.1, colors[index_colors[i]], 2, cv2.LINE_4)
	position = (position[0], position[1] + 40)
	cv2.putText(image, fonts[i].lower(), position, i, 1.1, colors[index_colors[i]], 2, cv2.LINE_4)
	position = (position[0], position[1] + 40)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'cv2.putText() using all OpenCV fonts')
