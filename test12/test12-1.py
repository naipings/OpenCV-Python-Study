#绘制函数中的shift参数
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
image = np.zeros((500, 500, 3), dtype="uint8")

# 修改画布背景颜色，这里设置为灰色
image[:] = colors['gray']

# 绘制两个半径为200的圆。其中之一使用shift=2的值来提供亚像素精度。
# 在这种情况下，应该将原点和半径都乘以因子4（2^shift=2）
# shift = 2
# factor = 2 ** shift
# print("factor: '{}'".format(factor))
# cv2.circle(image, (int(round(249.99 * factor)), int(round(249.99 * factor))), 200 * factor, colors['red'], 1, shift=shift)
# cv2.circle(image, (249, 249), 200, colors['green'], 1)

# 为了更方便的使用浮点坐标，也可以对绘图函数进行封装，
# 以cv2.circle()函数为例创建函数draw_float_circle()，它可以使用shift参数属性来处理浮点坐标：
def draw_float_circle(img, center, radius, color, thickness=1, lineType=8, shift=4):
	factor = 2 ** shift
	center = (int(round(center[0] * factor)), int(round(center[1] * factor)))
	radius = int(round(radius * factor))
	cv2.circle(img, center, radius, color, thickness, lineType, shift)
# 原本半径都是200，本人自我调整了一下，方便观察：	
draw_float_circle(image, (250, 250), 200, colors['red'], 1, 8, 0)
draw_float_circle(image, (249.9, 249.9), 210, colors['green'], 1, 8, 1)
draw_float_circle(image, (249.99, 249.99), 220, colors['blue'], 1, 8, 2)
draw_float_circle(image, (249.999, 249.999), 230, colors['yellow'], 1, 8, 3)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'cv2.circle()')