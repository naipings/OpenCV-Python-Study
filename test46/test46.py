# 轮廓识别
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 我们之前已经介绍了 cv2.approxPolyDP()，它可以使用 Douglas Peucker算法用较少的点来使一个轮廓逼近检测的轮廓。
# 此函数中的一个关键参数是 epsilon，其用于设置近似精度。
# 我们使用 cv2.approxPolyDP()，以便根据被抽取的轮廓中的检测到顶点的数量识别轮廓(例如，三角形，方形，矩形，五角形或六角形)。
# 为了减少点数，给定某个轮廓，我们首先计算轮廓的边( perimeter )。
# 基于边，建立 epsilon参数， epsilon 参数计算如下：
# epsilon = 0.03 * perimeter

# 如果该常数变大(例如，从 0.03 变为 0.1 )，则 epsilon参数也会更大，近似精度将减小，
# 这导致具有较少点的轮廓，并且导致顶点的缺失，对轮廓的识别也将不正确，因为它基于检测到的顶点的数量；
# 另一方面，如果该常数较小(例如，从0.03 变为 0.001)，则 epsilon参数也将变小，因此，近似精度将增加，
# 将产生具有更多点的近似轮廓，对轮廓的识别同样会出现错误，因为获得了虚假顶点。

# 构建测试图像
image = np.ones((300,700,3), dtype='uint8')
cv2.circle(image, (100, 80), 65, (64, 128, 0), -1)
pts = np.array([[300, 10], [400, 150], [200, 150]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(image, [pts], (64, 255, 64))
cv2.rectangle(image, (450, 20),(650, 150),(125, 125, 255),-1)
cv2.rectangle(image, (50, 180),(150, 280),(255, 125, 0),-1)
pts = np.array([[365, 220], [320, 282], [247, 258], [247, 182], [320, 158]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(image, [pts], (125, 64, 125))
pts = np.array([[645, 220], [613, 276], [548, 276], [515, 220], [547, 164],[612, 164]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(image, [pts], (255, 255, 0))

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
# 轮廓检测
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

image_contours = image.copy()
image_recognition_shapes = image.copy()

# 函数array_to_tuple()、draw_contour_points()、draw_contour_outline()，详见：test44.py
def array_to_tuple(arr):
    """将列表转换为元组"""
    return tuple(arr.reshape(1, -1)[0])

def draw_contour_points(img, cnts, color):
    """绘制所有检测到的轮廓点"""
    for cnt in cnts:
        squeeze = np.squeeze(cnt)
        for p in squeeze:
            pp = array_to_tuple(p)
            cv2.circle(img, pp, 10, color, -1)
    return img

def draw_contour_outline(img, cnts, color, thickness=1):
    """绘制所有轮廓"""
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

# 绘制所有检测的轮廓
draw_contour_outline(image_contours, contours, (255, 255, 255), 4)

def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """获取图形坐标中心点"""
    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)

def detect_shape(contour):
    """形状识别"""
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    contour_approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(contour_approx) == 3:
        detected_shape = 'triangle'
    elif len(contour_approx) == 4:
        x, y, width, height = cv2.boundingRect(contour_approx)
        aspect_ratio = float(width) / height
        if 0.90 < aspect_ratio < 1.10:
            detected_shape = "square"
        else:
            detected_shape = "rectangle"
    elif len(contour_approx) == 5:
        detected_shape = "pentagon"
    elif len(contour_approx) == 6:
        detected_shape = "hexagon"
    else:
        detected_shape = "circle"
    return detected_shape, contour_approx

for contour in contours:
    # 计算轮廓的矩
    M = cv2.moments(contour)
    # 计算轮廓的质心
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    # 识别轮廓形状
    shape, vertices = detect_shape(contour)
    # 绘制轮廓
    draw_contour_points(image_contours, [vertices], (255, 255, 255))
    # 将形状的名称置于形状的质心
    (x, y) = get_position_to_draw(shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    cv2.putText(image_recognition_shapes, shape, (x+35, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    """图像可视化"""
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')

# 图像进行阈值处理后，常见的输出是黑白图像
# 因此，为了更好的可视化效果，修改背景颜色
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('silver')
plt.suptitle("Functionality related to contours", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
show_img_with_matplotlib(image_contours, "contours outline (after approximation)", 3)
show_img_with_matplotlib(image_recognition_shapes, "contours recognition", 4)

plt.show()
