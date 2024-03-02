# 相机姿态估计
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 为了估计相机姿态，需要使用 cv2.aruco.estimatePoseSingleMarkers()函数，
# 它估计单个标记的姿态，姿态由旋转和平移向量组成，其用法如下：
# cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]]) -> rvecs, tvecs, _objPoints
# 其中，cameraMatrix 和 distCoeffs 分别是相机矩阵和畸变系数；它们是相机校准后获得的值；
# 参数corners 是一个向量，包含每个检测到的标记的四个角；
# markerLength参数是标记边的长度；返回的平移向量保存在同一单元中。
# 此函数返回每个检测到的标记的 rvecs(旋转向量)、tvecs(平移向量)和 _objPoints(所有检测到的标记角的对象点数组)。

# 标记坐标系以标记的中间为中心。因此，标记的四个角的坐标如下：
# (-markerLength/2, markerLength/2, 0)
# (markerLength/2, markerLength/2, 0)
# (markerLength/2, -markerLength/2, 0)
# (-markerLength/2, -markerLength/2, 0)

# Aruco还提供 cv2.aruco.drawAxis()函数，用于为每个检测到的标记绘制系统轴，其用法如下：
# cv2.aruco.drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length) -> image
# length参数设置绘制轴的长度（与 tvec单位相同），其它参数与之前函数相同。

# 加载相机校准数据
with open('../opencvStudy/test50/calibration.pckl', 'rb') as f:
    cameraMatrix, distCoeffs = pickle.load(f)
# 创建字典对象
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# 创建参数对象
parameters = cv2.aruco.DetectorParameters_create()

# 创建视频捕获对象
capture = cv2.VideoCapture(0)

while True:
    # 捕获视频帧
    ret, frame = capture.read()
    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 探测标记
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)
    # 绘制标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))
    # 绘制被拒绝候选标记
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))

    if ids is not None:
        # rvecs, tvecs分别是角点中每个标记的旋转和平移向量
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
        # 绘制系统轴
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1)

    # 绘制结果帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
