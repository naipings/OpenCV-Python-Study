# 椭圆
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

# 绘制椭圆的函数cv2.ellipse()的用法如下：
# cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=8, shift=0)
# 此函数用于绘制不同类型的椭圆：
# angle参数（以度为单位）可以旋转椭圆；axes参数控制长短轴的大小；startAngle和endAngle参数用于设置所需的椭圆弧（以度为单位），
# 例如，需要完整闭合的椭圆，则startAngle=0、endAngle=360：
cv2.ellipse(image, (100, 100), (60, 40), 0, 0, 360, colors['red'], -1)
cv2.ellipse(image, (100, 200), (80, 40), 0, 0, 360, colors['green'], 3)
cv2.ellipse(image, (100, 200), (10, 40), 0, 0, 360, colors['blue'], 3)
cv2.ellipse(image, (300, 300), (20, 80), 0, 0, 180, colors['yellow'], 3)
cv2.ellipse(image, (300, 100), (20, 80), 0, 0, 270, colors['cyan'], 3)
cv2.ellipse(image, (250, 250), (40, 40), 0, 0, 360, colors['magenta'], 3)
cv2.ellipse(image, (400, 100), (30, 60), 45, 0, 360, colors['rand'], 3)
cv2.ellipse(image, (400, 400), (30, 60), -45, 0, 360, colors['rand'], 3)
cv2.ellipse(image, (200, 400), (30, 60), -45, 0, 225, colors['rand'], -1)

# 调用show_with_matplotlib()函数：
show_with_matplotlib(image, 'cv2.ellipse()')
