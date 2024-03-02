# 相机校准
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 在使用检测到的标记获得相机姿态之前，需要知道相机的标定参数，
# ArUco提供了执行此任务所需的校准程序，校准程序仅需执行一次，因为程序执行过程中并未修改相机光学元件。
# 校准过程中使用的主要函数是 cv2.aruco.calibrateCameraCharuco()，其使用从板上提取的多个视图中的一组角来校准相机。
# 校准过程完成后，此函数返回相机矩阵(一个 3 x 3 浮点相机矩阵)和一个包含失真系数的向量，
# 3 x 3 矩阵对焦距和相机中心坐标(也称为内在参数)进行编码，而失真系数对相机产生的失真进行建模。

# 函数的用法如下：
# calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs
# 其中，charucoCorners 是一个包含检测到的 charuco 角的向量，
# charucoIds 是标识符列表，
# board 表示板布局，
# imageSize 是输入图像大小。
# 输出向量 rvecs 包含为每个板视图估计的旋转向量，
# tvecs 是为每个模式视图估计的平移向量，
# 此外，还需要返回相机矩阵 cameraMatrix 和失真系数 distCoeffs。

# 板是使用 cv2.aruco.CharucoBoard_create()函数创建的，其函数用法如下：(这里说的是OpenCV4.7x以下版本的用法)
# CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary) -> retval
# 这里，squareX 是 x 方向的方格数，
# squaresY 是 y 方向的方格数，
# squareLength 是棋盘方边长(通常以米为单位)，
# markerLength 是标记边长(与 squareLength 单位相同)，
# 以及dictionary 设置字典中要使用的第一个标记，以便在板内创建标记：
# 创建板:
# 4.7x以下Opencv版本：(使用本人pytorch虚拟环境，里面的opencv-python是4.5.3.56版本)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
board = cv2.aruco.CharucoBoard_create(3, 3, .025, .0125, dictionary)
img = board.draw((200 * 3, 200 * 3))

# 4.7x及以上OpenCV版本：(此部分代码有些问题，故仅供参考)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
# # 构造函数cv2.aruco.CharucoBoard_create已重命名为cv2.aruco.CharucoBoard，
# # 其参数列表略有更改——应该传入一个包含两个值（表示大小）的元组，而不是前两个整数参数squaresX和squaresY
# board = cv2.aruco.CharucoBoard((3, 3), .025, .0125, dictionary)
# # board = cv2.aruco.DetectorParameters(3, 3, .025, .0125, dictionary)
# img = board.draw((200 * 3, 200 * 3))

# 可视化（为了看看版的样式）
# def show_img_with_matplotlib(color_img, title, pos):
#     """图像可视化"""
#     # img_RGB = color_img[:, :, ::-1]
#     img_RGB = color_img[:, ::-1]
#     ax = plt.subplot(1, 1, pos)
#     plt.imshow(img_RGB)
#     plt.title(title, fontsize=6)
#     plt.axis('off')
# plt.suptitle("Aruco markers creation", fontsize=14, fontweight='bold')
# show_img_with_matplotlib(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), "", 1)
# plt.show()


# 该board稍后在校准过程中由cv2.aruco.calibrateCameraCharuco()函数使用：
# 创建字典对象
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# 创建视频捕获对象
cap = cv2.VideoCapture(0)

all_corners = []
all_ids = []
counter = 0
for i in range(300):
    # 读取帧
    ret, frame = cap.read()
    # 转化为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 标记检测
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
            all_corners.append(res2[1])
            all_ids.append(res2[2])
        # 绘制探测标记
        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    counter += 1

try:
    # 相机校准
    cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
except:
    cap.release()
    print("Calibration could not be done ...")

# 获取校准结果
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal

# 校准过程完成后，使用pickle将相机矩阵和失真系数保存到磁盘：
f = open('../opencvStudy/test50/calibration.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs), f)
f.close()

# 校准程序完成后，就可以执行相机姿态估计了。
