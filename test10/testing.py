# OpenCV 提供了许多绘制基本图形的函数，包括直线、矩形和圆形等；除此之外，使用 OpenCV，也可以绘制其它更多的基本图形。
# 图像上绘制基本形状有许多实用的场景，常见的用例主要包括：
# 1.显示算法的一些中间结果
# 2.显示算法的最终结果
# 3.显示一些调试信息

# 在学习如何绘制具有不同颜色的基本图形和文本之前，首先介绍不同颜色的构建方式：
# 我们可以构建颜色字典，使用颜色字典定义要使用的主要颜色。
# 参考：https://blog.csdn.net/LOVEmy134611/article/details/119712265
# 由上表可以构造颜色字典如下：
# colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'cyan': (255, 255, 0), 'magenta': (255, 0, 255), 'yellow': (0, 255, 255), 'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (125, 125, 125), 'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220), 'rand': np.random.randint(0, high=256, size=(3,)).tolist()}

# 以上字典中定义了一些预定义的颜色，如果要使用特定颜色，例如红色(red)：
# colors['red']
# 或者，可以使用 (0, 0, 255) 来得到红色。但是使用这个字典，不需要记住 RGB 颜色空间的色值，比数字三元组更容易使用。


# 除了使用字典外，另一种常见的方法是创建一个 colors_constant.py 文件来定义颜色。其中，每种颜色都由一个常量定义：
# BLUE = (255, 0, 0)
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# YELLOW = (0, 255, 255)
# MAGENTA = (255, 0, 255)
# CYAN = (255, 255, 0)
# LIGHT_GRAY = (220, 220, 220)
# DARK_GRAY = (50, 50, 50)

# 在项目目录的其他文件中，可以使用以下代码使能够引用这些常量：
# import colors_constant as colors
# print("red: {}".format(colors.RED))

# 此外，由于我们使用 Matplotlib 显示图形，因此我们需要通用函数 show_with_matplotlib()，
# 其带有两个参数的，第一个是要显示的图像，第二个是要图形窗口的标题。
# 因为必须使用 Matplotlib 显示彩色图像，因此首先需要将 BGR 图像转换为 RGB。
# 此函数的第二步也是最后一步是使用 Matplotlib 函数显示图像：
# def show_with_matplotlib(img, title):
#     # 将 BGR 图像转换为 RGB
#     img_RGB = img[:, :, ::-1]
#     # 使用 Matplotlib 显示图形
#     plt.imshow(img_RGB)
#     plt.title(title)
#     plt.show()

# 以下，演示colors常量和show_with_matplotlib()函数的使用：

import cv2
import numpy as np

# 不知道为什么，我的电脑运行这种加了import matplotlib.pyplot as plt文件的代码，就必须在前面加上如下两行，程序才能正常运行
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
# 创建一个大小为500*500的图像，具有三个通道（彩色通道），数据类型为uint8（8位无符号整数），原始背景为黑色：
image = np.zeros((500, 500, 3), dtype="uint8")

# 修改画布背景颜色
# 如果希望将背景颜色修改为随机数，则执行以下操作：
image[:] = colors['rand']

# 利用 colors 字典绘制直线
# 使用cv2.line()函数（后面会介绍）绘制一些直线，每条直线都使用字典填充颜色。
separation = 40
for key in colors:
    cv2.line(image, (0, separation), (500, separation), colors[key], 15)
    separation += 40

# 显示图形
# 最后，使用创建的show_with_matlotlib()函数绘制图像：
show_with_matplotlib(image, 'Dictionary with some predefined colors')

