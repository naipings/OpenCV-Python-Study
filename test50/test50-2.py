# 检测标记
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 可以使用cv2.aruco.detectMarkers()函数检测图像中的标记：
# corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)

# cv2.aruco.detectMarkers()函数的
# 第一个参数是要检测标记的灰度图像；
# 第二个参数是创建的字典对象；
# 第三个参数其它可以用于在检测过程中自定义的参数。

# 函数返回值如下：
# 1.返回检测到的标记的角列表，对于每个标记，返回其四个角(左上角、右上角、右下角和左下角)
# 2.返回检测到的标记的标识符列表
# 3.返回拒绝候选标记列表，它由所有找到的方块组成，但它们没有被适当的编码；每个被拒绝的候选标记同样由它的四个角组成

# 为了进行演示，我们检测摄像头中检测到的标记。首先，使用前面提到的 cv2.aruco.detectMarkers()函数检测标记，
# 然后，我们将使用 cv2.aruco.drawDetectedMarkers()函数绘制检测到的标记和拒绝的候选标记，如下所示：

# 4.7x以下Opencv版本：(使用本人pytorch虚拟环境，里面的opencv-python是4.5.3.56版本)
# # 创建字典对象
# aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# # 创建参数对象
# parameters = cv2.aruco.DetectorParameters_create()

# # 创建视频捕获对象
# capture = cv2.VideoCapture(0)

# while True:
#     # 捕获视频帧
#     ret, frame = capture.read()
#     # 转化为灰度图像
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 检测图像中标记
#     corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
#     # 绘制检测标记
#     frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
#     # 绘制被拒绝标记
#     frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))

#     # 展示结果
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# 4.7x及以上OpenCV版本：
# 创建字典对象
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
# 创建参数对象
parameters = cv2.aruco.DetectorParameters()

# 创建视频捕获对象
capture = cv2.VideoCapture(0)

while True:
    # 捕获视频帧
    ret, frame = capture.read()
    # 转化为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测图像中标记
    detector = cv2.aruco.ArucoDetector(aruco_dictionary,parameters)
    corners, ids, rejected_corners = detector.detectMarkers(gray_frame)
    # 绘制检测标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
    # 绘制被拒绝标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))

    # 展示结果
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 销毁窗口
capture.release()
cv2.destroyAllWindows()

# 执行程序后，可以看到检测到的标记用绿色边框绘制，而被拒绝的候选标记用红色边框绘制
