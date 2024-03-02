# 保存视频文件

# 1.计算帧率（FPS）————每秒处理的帧（画面）数，FPS越高越好。
# 但是，算法每秒应处理的帧数取决于需要解决的特定问题。

import cv2
import argparse
import time

capture = cv2.VideoCapture(0)

if capture.isOpened() is False:
    print("Error opening the camera")

while capture.isOpened():
    ret, frame = capture.read()
    if ret is True:
        processing_start = time.time()
        cv2.imshow("Input frame from the camera", frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale input camera', gray_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        processing_end = time.time()
        processing_time_frame = processing_end - processing_start
        print("FPS: {}".format(1.0 / processing_time_frame))
    else:
        break

capture.release()
cv2.destroyAllWindows()

# 上述脚本中，首先需要获取处理开始的时间：processing_start = time.time()
# 然后，获取所有处理完后的时间：processing_end = time.time()
# 接下来，计算时间差：processing_time_frame = processing_end - processing_start
# 最后，计算并打印FPS：print("FPS: {}".format(1.0 / processing_time_frame))

# 直接Run Code就行