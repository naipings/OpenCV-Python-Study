# 多边形
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

# 多边形绘制函数cv2.polylines()的用法如下：
# cv2.polylines(img, pts, isClosed, color, thickness=1, lineType=8, shift=0)
# 此函数用于绘制多边形。其中：
# 关键参数是pts，用于提供定义多边形的数组，这个参数的形状是（number_vertex, 1, 2），可以通过使用np.array创建坐标（np.int32类型）来定义它，
# 然后对其进行整形以适应pts参数所需形状。
# 如果要创建一个三角形，代码如下所示：
pts = np.array([[400, 5], [350, 200], [450, 200]], np.int32)
pts = pts.reshape((-1, 1, 2))
print("shape of pts '{}'".format(pts.shape))
cv2.polylines(image, [pts], True, colors['green'], 3)

#另一个重要的参数isClosed，如果此参数为True，则多边形将被绘制为闭合的；否则，第一个顶点和最后一个顶点之间的线段将不会被绘制，
# 从而产生开放的多边形：
pts = np.array([[400, 250], [350, 450], [450, 450]], np.int32)
pts = pts.reshape((-1, 1, 2))
print("shape of pts '{}'".format(pts.shape))
cv2.polylines(image, [pts], False, colors['green'], 3)

# 自我尝试，画一个矩形
pts = np.array([[100, 50], [100, 350], [250, 350], [250, 50]], np.int32)
pts = pts.reshape((-1, 1, 2))
print("shape of pts '{}'".format(pts.shape))
cv2.polylines(image, [pts], True, colors['blue'], 3)

# 调用show_with_matplotlib()函数：
show_with_matplotlib(image, 'cv2.polylines()')
