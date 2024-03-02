# 箭头
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

# 箭头绘制函数cv2.arrowedLine()的用法如下：
# cv2.arrowedLine(img, pt1, pt2, color, thickness=1, lineType=8, shift=0, tipLength=0.1)
# 此函数用于绘制箭头，箭头从pt1定义的点指向pt2定义的点。
# 箭头尖端的长度可以由tipLength()参数控制，该参数是根据线段长度（pt1和pt2之间的距离）的百分比定义的：
# 箭头尖端的长度为线段长度的 10%
cv2.arrowedLine(image, (50, 50), (450, 50), colors['cyan'], 3, 8, 0, 0.1)
# 箭头尖端的长度为线段长度的 30%
cv2.arrowedLine(image, (50, 200), (450, 200), colors['magenta'], 3, cv2.LINE_AA, 0, 0.3)
# 箭头尖端的长度为线段长度的 30%
cv2.arrowedLine(image, (50, 400), (450, 400), colors['blue'], 3, 8, 0, 0.3)

# 以上代码定义了三个箭头，除了箭头的大小不同外，使用了不同的lineType参数cv2.LINE_AA（也可以写16）和8（也可以写cv2.LINE_8），
# 调用show_with_matplotlib()函数后，可以观察它们之间的区别：
show_with_matplotlib(image, 'cv2.arrowedLine()')
