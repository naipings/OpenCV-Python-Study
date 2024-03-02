# 圆形
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
image = np.zeros((500, 500, 3), dtype="uint8")

# 修改画布背景颜色，这里设置为灰色
image[:] = colors['gray']

# 圆形图形绘制函数cv2.circle()，用法如下：
# img = cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
# 此函数以点center为中心绘制一个半径为radius的圆：
cv2.circle(image, (50, 50), 40, colors['magenta'], 3)
cv2.circle(image, (150, 150), 40, colors['rand'], -1)
cv2.circle(image, (250, 250), 50, colors['yellow'], 5)
cv2.circle(image, (250, 250), 60, colors['yellow'], 2)
cv2.circle(image, (350, 350), 40, colors['cyan'], -2)
cv2.circle(image, (450, 450), 40, colors['blue'], 3)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'cv2.circle()')