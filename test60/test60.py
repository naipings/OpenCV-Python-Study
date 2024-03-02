# 用dlib检测面部特征点
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import dlib

# 使用dlib检测面部特征点，需要首先下载检测面部特征点检测器文件（见文件夹Facial_feature_point_detection_detector）：
# 加载图像并转换为灰度图像
test_face = cv2.imread("../opencvStudy/test60/imgs/test2.jpg")
test_face = cv2.resize(test_face, None, fx=0.2, fy=0.2)
gray = cv2.cvtColor(test_face, cv2.COLOR_BGR2GRAY)

# 人脸检测
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 0)

# 检测面部特征点
p = "../opencvStudy/test60/Facial_feature_point_detection_detector/shape_predictor_68_face_landmarks.dat"
# 第二种面部特征点检测方法，第一行代码：
predictor = dlib.shape_predictor(p)


# 该方法的第二行代码中的 shape 变量是一个 dlib_full_object_detection 对象，用于表示图像中对象的位置，
# 接下来需要将其转换为numpy数组，编写shape_to_np() 函数执行此转换：
def shape_to_np(dlib_shape, dtype="int"):
    # 初始化 (x, y) 坐标列表
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)
    # 循环所有面部特征点，并将其转换为 (x, y) 坐标的元组
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)
    # 返回 (x,y) 坐标的列表
    return coordinates

# show_detection() 用于显示检测结果（绘制检测到的人脸边框）：
def show_detection(image, faces):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 1)
    return image


# 接下来，介绍几种常用的绘制检测到的面部特征点的不同方法：
# 1.使用线条绘制面部特征点形状，以连接绘制脸部的不同部分的轮廓（例如鼻子、眼睛等）：
# 定义不同特征点取值切片
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
ALL_POINTS = list(range(0, 68))
# 使用线条绘制面部特征点
def draw_shape_lines_all(np_shape, image):
    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)
# 连接不同的点来绘制曲线形状
def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=2, lineType=cv2.LINE_8)


# 2.调用 draw_shape_lines_range() 函数，可以仅绘制指定面部组件轮廓线，例如下颚线JAWLINE_POINTS：
# draw_shape_lines_range(shape, test_face, JAWLINE_POINTS)


# 3.绘制所有特征点及其位序：
# 绘制指定的特征点
def draw_shape_points_pos_range(np_shape, image, points):
    np_shape_display = np_shape[points]
    draw_shape_points_pos(np_shape_display, image)
# 使用每个特征点及其位序绘制形状
def draw_shape_points_pos(np_shape, image):
    for idx, (x, y) in enumerate(np_shape):
        # 绘制每个检测到的特征点的位序
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        # 在每个特征点位置上绘制一个点
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
# 可以用两种方法来绘制检测到的所有特征点及其位序（见下面）


# 4.我们也可以仅绘制所有特征点：
# 绘制指定特征点
def draw_shape_points_range(np_shape, image, points):
    np_shape_display = np_shape[points]
    draw_shape_points(np_shape_display, image)
# 绘制所有特征点
def draw_shape_points(np_shape, image):
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


# 5.利用上述函数，我们可以仅绘制指定特征点，例如仅绘制眼部和鼻子特征点：
# draw_shape_points_pos_range(shape, test_face, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)


# dlib还提供了检测与双眼和鼻尖位置相对应的5个面部特征点检测器，如果想要使用此检测器，需要首先下载配置器文件，然后加载它：
# p = "shape_predictor_5_face_landmarks.dat"


# 第二种面部特征点检测方法，第二行代码：
for rect in rects:
    shape = predictor(gray, rect)

shape = shape_to_np(shape, int)

# 绘制检测框
test_face = show_detection(test_face.copy(), rects)

# 最后，在图像中绘制68个面部特征点：
# 方法1：
# draw_shape_lines_all(shape, test_face)

# 方法2：
# draw_shape_lines_range(shape, test_face, JAWLINE_POINTS)

# 方法3中，可以用两种方法来绘制检测到的所有特征点及其位序：
# 方法3.1：
# draw_shape_points_pos(shape, test_face)
# 方法3.2：
# draw_shape_points_pos_range(shape, test_face, ALL_POINTS)

# 方法4中，可以用两种方法绘制所有特征点：
# 方法4.1：
# draw_shape_points(shape, test_face)
# 方法4.2：
# draw_shape_points_range(shape, test_face, ALL_POINTS)

# 方法5：
draw_shape_points_pos_range(shape, test_face, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)

cv2.imshow("Landmarks detection using dlib", test_face)
cv2.waitKey(0)
cv2.destroyAllWindows()