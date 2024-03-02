# 轮廓检测
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 接下来将介绍如何在OpenCV中检测轮廓。为此我们首先绘制一些预定义的形状，然后使用绘制的形状进行轮廓检测：
def build_sample_image():
    """绘制一些基本形状"""
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (50, 50), (250, 250), (255, 0, 255), -1)
    cv2.rectangle(img, (100, 100), (200, 200), (70, 70, 70), -1)
    cv2.circle(img, (350, 350), 100, (255, 255, 0), -1)
    cv2.circle(img, (350, 350), 50, (70, 70, 70), -1)
    return img
    
# 加载图像并转换为灰度图像
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用 cv2.threshold() 函数获取二值图像
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

# 上述函数绘制了两个填充的矩形和两个填充的圆圈，
# 此函数创建的图像具有两个外部轮廓和两个内部轮廓，并在加载图像后，将其转换为灰度图形，并获取二值图像，
# 此二值图像将用于使用 cv2.findContours()函数查找轮廓。
# 接下来，就可以调用 cv2.findContours()检测到 利用build_sample_image()函数创建的图形的轮廓，

# cv2.findContours() 函数用法如下：
# cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
# 其中，method参数设置检索与每个检测到的轮廓相关的点时使用的近似方法，
# cv2.findCOntours()返回检测到的二值图像中的轮廓（例如，经过阈值处理后得到的图像），
# 每个轮廓包含定义边界的所有轮廓点，检索到的轮廓可以以不同的模式（mode）输出：（输出模式具体可见：教程网址，或者截屏图片）

# 轮廓检测
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 打印使用不同 mode 参数获得的轮廓数
print("detected contours (RETR_EXTERNAL): '{}' ".format(len(contours)))
print("detected contours (RETR_LIST): '{}' ".format(len(contours2)))
print("detected contours (RETR_TREE): '{}' ".format(len(contours3)))

image_contours = image.copy()
image_contours_2 = image.copy()

# 查看轮廓外观（详见test38.py）
def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

# 绘制检测到的轮廓
draw_contour_outline(image_contours, contours, (0, 0, 255), 5)
draw_contour_outline(image_contours_2, contours2, (255, 0, 0), 5)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
show_img_with_matplotlib(image_contours, "contours (RETR EXTERNAL)", 3)
show_img_with_matplotlib(image_contours_2, "contours (RETR LIST)", 4)

plt.show()