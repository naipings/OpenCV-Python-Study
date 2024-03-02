# 轮廓筛选
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 如果想要计算检测到的轮廓的大小，可以使用基于图像矩的方法或使用OpenCV函数cv2.contourArea()来计算检测到的轮廓的大小，
# 接下来，让我们将根据每个检测到的轮廓大小对其进行排序，
# 在实践中，某些小的轮廓可能是噪声导致的，可能需要对轮廓进行筛选。

# 首先在画布上绘制不同半径的圆，用于后续检测：
# 画布
image = np.ones((300,700,3), dtype='uint8')
# 绘制不同半径的圆
cv2.circle(image, (20, 20), 8, (64, 128, 0), -1)
cv2.circle(image, (60, 80), 25, (128, 255, 64), -1)
cv2.circle(image, (100, 180), 50, (64, 255, 64), -1)
cv2.circle(image, (200, 250), 45, (255, 128, 64), -1)
cv2.circle(image, (300, 250), 30, (35, 128, 35), -1)
cv2.circle(image, (380, 100), 15, (125, 255, 125), -1)
cv2.circle(image, (600, 210), 55, (125, 125, 255), -1)
cv2.circle(image, (450, 150), 60, (0, 255, 125), -1)
cv2.circle(image, (330, 180), 20, (255, 125, 0), -1)
cv2.circle(image, (500, 60), 35, (125, 255, 0), -1)
cv2.circle(image, (200, 80), 65, (125, 64, 125), -1)
cv2.circle(image, (620, 80), 48, (255, 200, 128), -1)
cv2.circle(image, (400, 260), 28, (255, 255, 0), -1)

# 接下来，检测图中轮廓：
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 阈值处理
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
# 检测轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 打印检测到的轮廓数
print("detected contours: '{}' ".format(len(contours)))

# 根据每个检测到的轮廓大小进行排序：
def sort_contours_size(cnts):
    """根据大小对轮廓进行排序"""

    cnts_sizes = [cv2.contourArea(contour) for contour in cnts]
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, cnts)))
    return cnts_sizes, cnts
    
(contour_sizes, contours) = sort_contours_size(contours)

# 见：test46.py
def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """获取图形坐标中心点"""
    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)

# 最后可视化：
def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

show_img_with_matplotlib(image, 'image', 1)

for i, (size, contour) in enumerate(zip(contour_sizes, contours)):
    # 计算轮廓的矩
    M = cv2.moments(contour)
    # 质心
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    # get_position_to_draw() 函数与上例相同
    (x, y) = get_position_to_draw(str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, 5)

    # 将排序结果置于形状的质心
    cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

show_img_with_matplotlib(image, "result", 2)

plt.show()
