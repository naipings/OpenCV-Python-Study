# 使用基于dlib DCF的跟踪器进行对象跟踪
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import dlib

# 除了人脸外，基于 dlib DCF 的跟踪器可以用于跟踪任意对象。
# 接下来，我们使用鼠标选择要跟踪的对象，并监听键盘事件，
# 如果按 1，将开始跟踪预定义边界框内的对象；
# 如果按 2，预定义的边界框将被清空，跟踪算法将停止，并等待用户选择另一个边界框。

# 例如，我们对检测小姐姐并不感兴趣，而更喜欢猫，那么我们可以首先用鼠标绘制矩形框选择喵咪，然后按 1 开始追踪小猫咪，
# 如果我们想要追踪其他物体，可以按 2 重新绘制矩形框并进行追踪。

def draw_text_info():
    # 绘制文本的位置
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)
    menu_pos_3 = (10, 60)
    # 菜单项
    info_1 = "Use left click of the mouse to select the object to track"
    info_2 = "Use '1' to start tracking, '2' to reset tracking and 'q' to exit"

    # 绘制菜单信息
    cv2.putText(frame, "Use '1' to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, info_2, menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_state:
        cv2.putText(frame, "tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "not tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# 用于保存要跟踪的对象坐标的结构
points = []

def mouse_event_handler(event, x, y, flags, param):
    # 对全局变量的引用
    global points
    # 添加要跟踪的对象的左上角坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
    # 添加要跟踪的对象的右下角坐标：
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))
# 创建视频捕获对象
capture = cv2.VideoCapture(0)

# 窗口名
window_name = "Object tracking using dlib correlation filter algorithm"
# 创建窗口
cv2.namedWindow(window_name)
# 绑定鼠标事件
cv2.setMouseCallback(window_name, mouse_event_handler)

# 初始化跟踪器
tracker = dlib.correlation_tracker()
tracking_state = False

while True:
    # 捕获视频帧
    ret, frame = capture.read()
    # 绘制菜单项
    draw_text_info()

    # 设置并绘制一个矩形，跟踪矩形框内的对象
    if len(points) == 2:
        cv2.rectangle(frame, points[0], points[1], (0, 0, 255), 3)
        dlib_rectangle = dlib.rectangle(points[0][0], points[0][1], points[1][0], points[1][1])

    if tracking_state is True:
        # 更新跟踪器并打印测量跟踪器的置信度
        print(tracker.update(frame))
        # 获取被跟踪对象的位置
        pos = tracker.get_position()
        # 绘制被跟踪对象的位置
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    # 捕获键盘事件
    key = 0xFF & cv2.waitKey(1)

    # 按下 1 键，开始追踪
    if key == ord("1"):
        if len(points) == 2:
            # Start tracking:
            tracker.start_track(frame, dlib_rectangle)
            tracking_state = True
            points = []

    # 展示结果图像
    cv2.imshow(window_name, frame)

    # 按下 2 键，停止跟踪
    if key == ord("2"):
        points = []
        tracking_state = False
    # 按下 q 键，返回
    if key == ord('q'):
        break

# 释放资源
capture.release()
cv2.destroyAllWindows()
