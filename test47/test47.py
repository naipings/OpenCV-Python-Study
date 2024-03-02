# 轮廓匹配
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Hu矩不变性可用于对象匹配和识别，我们将介绍如何基于 Hu矩不变性的匹配轮廓。
# OpenCV提供 cv2.matchShapes()函数可用于比较两个轮廓，其包含三种匹配算法，包括
# cv2.CONTOURS_MATCH_I1，cv2.CONTOURS_MATCH_I2 和 cv2.CONTOURS_MATCH_I3，这些算法都使用Hu矩不变性。
# 如果 A表示第一个对象，B表示第二个对象，则使用以下公式计算匹配性：（注：公式见网址教程，或者截屏图片）

# 接下来，我们使用 cv2.matchShapes()来计算轮廓与给定圆形轮廓的匹配程度。
# 首先，通过使用 cv2.circle()在图像中绘制圆形作为参考图像。
# 之后，加载绘制了不同形状的图像，然后在上述图像中查找轮廓：
def build_circle_image():
    """绘制参考圆"""
    img = np.zeros((500, 500, 3), dtype="uint8")
    cv2.circle(img, (250, 250), 200, (255, 255, 255), 1)
    return img
# 加载图像
image = cv2.imread("../opencvStudy/test47/imgs/test15.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_circle = build_circle_image()
gray_image_circle = cv2.cvtColor(image_circle, cv2.COLOR_BGR2GRAY)

# 二值化图像
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY_INV)
ret, thresh_circle = cv2.threshold(gray_image_circle, 70, 255, cv2.THRESH_BINARY)

# 检测轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_circle, hierarchy_2 = cv2.findContours(thresh_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

result_1 = image.copy()
result_2 = image.copy()
result_3 = image.copy()

# 见：test46.py
def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """获取图形坐标中心点"""
    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)

for contour in contours:
    # 计算轮廓的矩
    M = cv2.moments(contour)

    # 计算矩的质心
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # 使用三种匹配模式将每个轮廓与圆形轮廓进行匹配
    ret_1 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I1, 0.0)
    ret_2 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I2, 0.0)
    ret_3 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I3, 0.0)

    # 将获得的分数写在结果图像中
    (x_1, y_1) = get_position_to_draw(str(round(ret_1, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_2, y_2) = get_position_to_draw(str(round(ret_2, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_3, y_3) = get_position_to_draw(str(round(ret_3, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)

    cv2.putText(result_1, str(round(ret_1, 3)), (x_1+10, y_1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(result_2, str(round(ret_2, 3)), (x_2+10, y_2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_3, str(round(ret_3, 3)), (x_3+10, y_3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Functionality related to contours", fontsize=14, fontweight='bold')

show_img_with_matplotlib(result_1, "image", 1)
show_img_with_matplotlib(result_2, "threshold = 100", 2)
show_img_with_matplotlib(result_3, "contours outline (after approximation)", 3)

plt.show()