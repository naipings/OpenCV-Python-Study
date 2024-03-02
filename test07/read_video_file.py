# 读取视频文件

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video_path", help="path to the video file")
args = parser.parse_args()
capture = cv2.VideoCapture(args.video_path)
if capture.isOpened() is False:
    print("Error opening the video file!")

while capture.isOpened():
    ret, frame = capture.read()
    if ret is True:
        cv2.imshow('Original frame from the video file', frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale frame', gray_frame)
        # 按下 q 键可退出程序执行
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()

# 终端执行：python test07/read_video_file.py ../opencvStudy/test07/test.mp4

