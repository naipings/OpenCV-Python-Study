# 读取相机画面

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("index_camera", help="index of the camera to read from", type=int)
args = parser.parse_args()

capture = cv2.VideoCapture(args.index_camera) 
if capture.isOpened()is False:
    print("Error opening the camera")
while capture.isOpened():
    # 逐帧获取相机画面
    ret, frame = capture.read() # 从相机逐帧捕获画面，需要调用capture.read()方法，该方法从相机返回帧，同时还将返回一个布尔值，此布尔值指示是否已从对象（本例即：capture）正确读取帧

    if ret is True:
        # 显示捕获的帧画面
        cv2.imshow('Input frame from the camera', frame)
        # 将从相机捕获的帧转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 显示灰度帧图像
        cv2.imshow('Grayscale input camera', gray_frame)
        # 按下 q 键可退出程序执行
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()

# cv2.VideoCapture所必需的参数是index_camera，它指示要读取的相机的索引。
# 如果有一个摄像头连接到计算机，则它的索引为0，如果有第二台摄像头，可以通过传递参数值1来选择它，以此类推；需要注意的是，该参数的类型是int

# 执行程序，终端输入：python test05/read_camera.py 0
