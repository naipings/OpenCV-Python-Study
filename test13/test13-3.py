# 其他与文本相关的函数

# 第一个：cv2.getFontScaleFromHeight();
# 此函数返回字体比例（这是在cv2.putText()函数中使用的参数fontScale），以得到提供的高度（以像素为单位）并考虑字体类型（fontFace）和thickness
# retval = cv2.getFontScaleFromHeight(fontFace, pixelHeight, thickness=1)

# 第二个：cv2.getTextSize();
# 此函数根据以下参数：要绘制的text、字体类型（fontFace）、fontScale和thickness，获取文本大小（宽度和高度）。
# 此函数返回size和baseLine（它对应于相当于文本底部的基线的y坐标）：
# retval, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)

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
image = np.zeros((1200, 1200, 3), dtype="uint8")

# 修改画布背景颜色，这里设置为灰色
image[:] = colors['gray']

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.5
thickness = 5
text = 'abcdefghijklmnopqrstuvwxyz'
circle_radius = 10
ret, baseline = cv2.getTextSize(text, font, font_scale, thickness)
text_width, text_height = ret
text_x = int(round((image.shape[1] - text_width) / 2))
text_y = int(round((image.shape[0] + text_height) / 2))
cv2.circle(image, (text_x, text_y), circle_radius, colors['green'], -1)
cv2.rectangle(image, (text_x, text_y + baseline), (text_x + text_width - thickness, text_y - text_height), colors['blue'], thickness)
cv2.circle(image, (text_x, text_y + baseline), circle_radius, colors['red'], -1)
cv2.circle(image, (text_x + text_width - thickness, text_y - text_height), circle_radius, colors['cyan'], -1)
cv2.line(image, (text_x, text_y + int(round(thickness/2))), (text_x + text_width - thickness, text_y + int(round(thickness/2))), colors['yellow'], thickness)
cv2.putText(image, text, (text_x, text_y), font, font_scale,colors['magenta'], thickness)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'cv2.getTextSize() + cv2.putText()')
