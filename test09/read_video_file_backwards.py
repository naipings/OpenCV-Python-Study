# 视频属性的使用——视频的反向播放

# 该脚本使用一些前面所示的属性加载视频并反向播放（首先显示视频的最后一帧，然后播放倒数第二帧，以此类推）
# 为了实现此目的，需要使用属性：cv2.CAP_PROP_COUNT和cv2.CAP_PROP_POS_FRAMES

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video_path", help='path to the video fiel')
args = parser.parse_args()

capture = cv2.VideoCapture(args.video_path)

if capture.isOpened() is False:
    print("Error opening vieo stream or file")

# 第一步是获取最后一帧的索引：
frame_index = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1
print("Starting in frame: '{}'".format(frame_index))

while capture.isOpened() and frame_index >= 0:
    # 然后，将当前帧设置为所获取的位置：
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # 这样，就可以读取所获取的帧了：
    ret, frame = capture.read()

    if ret:
        cv2.imshow('Original frame', frame)
        # 最后，索引值减一以从视频文件中读取下一帧：
        frame_index = frame_index - 1
        print("next index to read: '{}'".format(frame_index))
 
        # Press q on keyboard to exit the program:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

capture.release()
cv2.destroyAllWindows()

# 终端执行：python test09/read_video_file_backwards.py ../opencvStudy/test09/test.mp4
