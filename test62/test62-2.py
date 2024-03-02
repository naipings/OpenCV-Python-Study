# 使用基于dlib DCF的跟踪器进行人脸跟踪————
# 2.对计算机相机中的人脸进行跟踪：
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import dlib

# 在进行人脸跟踪时，我们首先使用dlib人脸检测器进行初始化，然后使用基于dlib DCF的跟踪器DSST进行人脸跟踪。
# 调用以下函数初始化相关跟踪器：
tracker = dlib.correlation_tracker()

# 初始化变量tracking_face为False
tracking_face = False

# 使用 tracker.start_track() 可以开始跟踪。
# 在开始跟踪前，我们需要先执行人脸检测，并将检测到的人脸位置传递给这个方法：
# 从dlib上加载人脸检测器：
detector = dlib.get_frontal_face_detector()

# 创建视频捕获对象
capture = cv2.VideoCapture(0)

while True:
    # 捕获视频帧
    ret, frame = capture.read()
    if ret is True:        
        # 执行人脸检测，并将检测到的人脸位置传递给这个方法：
        if tracking_face is False:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 尝试检测人脸以初始化跟踪器
            # 执行检测：
            rects = detector(gray, 0)
            # 检查是否检测到人脸
            if len(rects) > 0:
                # 开始追踪
                tracker.start_track(frame, rects[0])
                tracking_face = True
        
        # 当检测到人脸后，人脸跟踪器将开始跟踪边界框内的内容。为了更新被跟踪对象的位置，需要调用 tracker.update() 方法：
        if tracking_face is True:
            tracker.update(frame)
            # tracker.update()方法更新跟踪器并返回衡量跟踪器置信度的指标，此指标可用于使用人脸检测重新初始化跟踪器。

            # 获取被跟踪对象的位置，需要调用 tracker.get_position() 方法：
            pos = tracker.get_position()
            # tracker.get_position()方法返回z被跟踪对象的位置。

            # 最后，绘制人脸的预测位置：
            cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)
    
        # 绘制结果帧
        cv2.imshow('frame', frame)

        # 捕获键盘事件
        key = 0xFF & cv2.waitKey(20)
        # 按 1 初始化追踪器
        if key == ord("1"):
            tracking_face = False
        # 按下 q 键可退出程序执行
        if key == ord('q'):
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()