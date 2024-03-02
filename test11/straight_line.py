# 注：OpenCV提供的大多数绘图函数都有共同的参数，
# 介绍请看：https://blog.csdn.net/LOVEmy134611/article/details/119712265

# 直线
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

# 直线绘制函数cv2.line()，函数用法如下：
# img = cv2.line(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)
# 此函数在img图像上画一条连接pt1和pt2的直线：
cv2.line(image, (0, 0), (500, 500), colors['magenta'], 3)
cv2.line(image, (0, 500), (500, 0), colors['cyan'], 10)
cv2.line(image, (250, 0), (250, 500), colors['rand'], 3)
cv2.line(image, (0, 250), (500, 250), colors['yellow'], 10)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'cv2.line()')