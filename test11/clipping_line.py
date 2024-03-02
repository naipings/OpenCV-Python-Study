# 剪裁线
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

# 剪裁线绘制函数cv2.circle()，使用方法如下：
# retval, pt1_new, pt2_new = cv2.clipLine(imgRect, pt1, pt2)
# cv2.circle()函数返回矩形内的线段（由输出点pt1_new和pt2_new定义），该函数根据定义的矩形imgRect裁剪线段。
# 如果两个原始点pt1和pt2都在矩形之外，则retval为False；否则返回True：
cv2.line(image, (0, 0), (500, 500), colors['green'], 3)
cv2.rectangle(image, (100, 100), (300, 300), colors['blue'], 3)
ret, p1, p2 = cv2.clipLine((100, 100, 300, 300), (0, 0), (300, 300))
if ret:
    cv2.line(image, p1, p2, colors['magenta'], 3)
ret, p1, p2 = cv2.clipLine((100, 100, 300, 300), (250, 150), (0, 400))
if ret:
    cv2.line(image, p1, p2, colors['cyan'], 3)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'cv2.clipLine()')