# 2.写入视频文件的流程

# 注意事项：https://blog.csdn.net/LOVEmy134611/article/details/119582351

# 下面演示如何创建视频文件：
# 使用cv2.VideoWriter()创建视频文件：
# 在创建的名为video_example.avi视频中，FOURCC值为XVID，视频文件格式为AVI(*.avi)，同时最后，应设置视频每一帧的FPS和尺寸。

import cv2
import argparse

# 必需的参数是输出视频文件名（例如：video_example.avi）：
parser = argparse.ArgumentParser()
parser.add_argument("output_video_path", help="path to the video file to write")
args = parser.parse_args()

# 要从连接到计算机的第一台相机拍摄视频画面。因此，首先创建对象：
capture = cv2.VideoCapture(0)

# 接下来，从capture对象中获取一些关键属性（帧宽度、帧高度和FPS），用于创建视频文件时使用：
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# 然后使用四字节编码FOURCC指定视频编解码器，此处将编解码器定义为XVID：
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 也可以使用以下方式指定视频编码器：
# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

# 然后，使用与输入相机相同的属性创建cv2.VideoWrite对象out_gray，
# cv2.VideoWrite的最后一个参数值三False表示以灰度方式写入视频。
# 如果我们想创建彩色视频，最后一个参数为True：
out_gray = cv2.VideoWriter(args.output_video_path, fourcc, int(fps), (int(frame_width), int(frame_height)), False)

# 使用capture.read()从capture对象获取相机帧，每一帧都被转换成灰度并写入视频文件，如果按下q键，程序结束：
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_gray.write(gray_frame)

        cv2.imshow('gray', gray_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 最后，释放所有内容（包括cv2.VideoCapture和cv2.VideoWriter对象，并销毁创建的窗口）：
capture.release()
out_gray.release()
cv2.destroyAllWindows()

#终端输入：python test08/write_video_file.py ../opencvStudy/test08/video_example.avi
