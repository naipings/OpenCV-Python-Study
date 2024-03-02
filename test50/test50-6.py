# 增强现实进阶
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 可以轻松修改test50-5.py的代码以融合更高级的数字信息，
# 接下来，我们将融合图像：

# 为了实现上述增强现实，实现draw_augmented_overlay()函数如下：
def draw_augmented_overlay(pts_1, overlay_image, image):
    """ 增强现实 """

    # 定义要绘制的叠加图像的正方形
    pts_2 = np.float32([[0, 0], [overlay_image.shape[1], 0], [overlay_image.shape[1], overlay_image.shape[0]],
                        [0, overlay_image.shape[0]]])
    # 绘制边框以查看图像边框
    cv2.rectangle(overlay_image, (0, 0), (overlay_image.shape[1], overlay_image.shape[0]), (255, 255, 0), 10)

    # 创建转换矩阵
    M = cv2.getPerspectiveTransform(pts_2, pts_1)

    # 使用变换矩阵M变换融合图像
    dst_image = cv2.warpPerspective(overlay_image, M, (image.shape[1], image.shape[0]))

    # 创建掩码
    dst_image_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(dst_image_gray, 0, 255, cv2.THRESH_BINARY_INV)
    # 使用计算出的掩码计算按位与
    image_masked = cv2.bitwise_and(image, image, mask=mask)
    # 两个图像进行加和创建结果图像
    result = cv2.add(dst_image, image_masked)
    return result
# draw_augmented_overlay()函数首先定义叠加图像的正方形。
# 然后计算变换矩阵，用于变换叠加图像得到 dst_image图像；
# 接下来，创建掩码并使用之前创建的掩码按位运算以获得 image_masked图像；
# 最后将 dst_image 和image_masked 相加，得到结果图像，并返回。


OVERLAY_SIZE_PER = 1
# 加载相机校准数据
with open('../opencvStudy/test50/calibration.pckl', 'rb') as f:
    cameraMatrix, distCoeffs = pickle.load(f)
# 创建字典对象
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
# 创建参数对象
parameters = cv2.aruco.DetectorParameters_create()

# 加载图像
image = cv2.imread('../opencvStudy/test50/imgs/test01.jpeg')
image = cv2.resize(image, None, fx=0.01, fy=0.01)
# cv2.imshow("", image)

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
            # 绘制图像
            draw_augmented_overlay(projected_desired_points, image, frame)

    # 绘制结果帧
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break