# 增强现实初探
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 使用draw_points()函数用于绘制这些点：
def draw_points(img, pts):
    """ 在图中绘制点 """
    pts = np.int32(pts).reshape(-1, 2)
    img = cv2.drawContours(img, [pts], -1, (255, 255, 0), -3)
    for p in pts:
        cv2.circle(img, (p[0], p[1]), 5, (255, 0, 255), -1)
    return img

# 在50.4.py之后，我们就可以在现实图像中融合一些图像、形状或3D模型，以展示完整的增强现实应用程序，
# 在第一个实例中，我们用标记的大小覆盖一个矩形。执行此功能的代码如下：

# 首先需要先定义想要覆盖的图像或模型的点，由于想将矩形覆盖在检测到的标记上，因此定义矩形坐标：
# desired_points = np.float32(
#                 [[-1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0], [1 / 2, -1 / 2, 0], [-1 / 2, -1 / 2, 0]]) * OVERLAY_SIZE_PER
# 需要注意的是，我们必须在标记坐标系（标记中心为坐标原点）中定义这些坐标，然后使用 cv2.projectPoints()函数投影这些点：
# projected_desired_points, jac = cv2.projectPoints(desired_points, rvecs, tvecs, cameraMatrix, distCoeffs)

# 最后，使用draw_points()函数绘制这些点。

OVERLAY_SIZE_PER = 1
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

    if ids is not None:
        # rvecs, tvecs分别是角点中每个标记的旋转和平移向量
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            # 定义要覆盖图像的点
            desired_points = np.float32(
                [[-1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0], [1 / 2, -1 / 2, 0], [-1 / 2, -1 / 2, 0]]) * OVERLAY_SIZE_PER
            # 投影点
            projected_desired_points, jac = cv2.projectPoints(desired_points, rvecs, tvecs, cameraMatrix, distCoeffs)
            # 绘制投影点
            draw_points(frame, projected_desired_points)

    # 绘制结果帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
