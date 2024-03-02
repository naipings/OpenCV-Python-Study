# 绘制函数中的lineType参数
# 参见：https://blog.csdn.net/LOVEmy134611/article/details/119712265

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

# 创建画布
image = np.zeros((20, 20, 3), dtype="uint8")

# 修改画布背景颜色，这里设置为灰色
image[:] = colors['gray']

# lineType可以采用三个不同的值。
# 为了更清晰地看到它们之间的差异，可以绘制三条具有相同粗细和倾斜度的线，仅lineType参数值不同：
# 在图片中，yellow=cv2.LINE_4、red=cv2.LINE_AA、green=cv2.LINE_8，可以清楚地看到使用三种不同线型绘制线条时的区别。
cv2.line(image, (5, 0), (20, 15), colors['yellow'], 1, cv2.LINE_4)
cv2.line(image, (0, 0), (20, 20), colors['red'], 1, cv2.LINE_AA)
cv2.line(image, (0, 5), (15, 20), colors['green'], 1, cv2.LINE_8)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'LINE_4  LINE_AA  LINE_8')
