# 绘制文本

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

# cv2.putText()函数的用法如下：
# img = cv.putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType= 8, bottomLeftOrigin=False)
# 此函数使用fontFace声明字体类型和fontScale因子从org坐标（如果bottomLeftOrigin=False则为左上角，否则为左下角）开始绘制提供的文本字符串text。
# 最后提供的参数lineType同样可以使用三个不同的可选值：cv2.LINE_4、cv2.LINE_8、cv2.LINE_AA：
cv2.putText(image, 'learn OpenCV-Python together', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors['cyan'], 2, cv2.LINE_4)
cv2.putText(image, 'learn OpenCV-Python together', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors['cyan'], 2, cv2.LINE_8)
cv2.putText(image, 'learn OpenCV-Python together', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors['cyan'], 2, cv2.LINE_AA)

# 调用show_with_matplotlib()函数显示图像：
show_with_matplotlib(image, 'cv2.putText()')
