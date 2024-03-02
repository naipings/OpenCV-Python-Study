# 具体：https://blog.csdn.net/LOVEmy134611/article/details/119582351

# 获取视频对象的属性

# 创建read_video_file_all_properties.py脚本来显示所有属性。
# 其中一些属性仅在使用相机时才有效（而在视频文件时无效）

# 为了防止下面print部分的代码闪红（报错），故加入这部分代码（仅为美观，无法构成完整程序）
import cv2
capture = cv2.VideoCapture(0)

# 首先创建decode_fourcc()函数，它将capture.get(cv2.CAP_PROP_FOURCC)返回的int类型的值转换为表示编解码器的字符串值，来正确输出编解码器：
def decode_fourcc(fourcc):
    fourcc_int = int(fourcc)

    print("int value of fourcc: '{}'".format(fourcc_int))

    fourcc_decode = ""
    for i in range(4):
        int_value = fourcc_int >> 8 * i & 0xFF
        print("int_value: '{}'".format(int_value))
        fourcc_decode += chr(int_value)
    
    return fourcc_decode

# decode_fourcc()工作原理、视频文件的主要属性和解释，均见上面网址。

# 使用以下代码，可以获取并打印所有属性：
print("CV_CAP_PROP_FRAME_WIDTH:'{}'".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("CV_CAP_PROP_FRAME_HEIGHT :'{}'".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
print("CAP_PROP_POS_MSEC :'{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
print("CAP_PROP_POS_FRAMES :'{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))
print("CAP_PROP_FOURCC :'{}'".format(decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))))
print("CAP_PROP_FRAME_COUNT :'{}'".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
print("CAP_PROP_MODE : '{}'".format(capture.get(cv2.CAP_PROP_MODE)))
print("CAP_PROP_BRIGHTNESS :'{}'".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))
print("CAP_PROP_CONTRAST :'{}'".format(capture.get(cv2.CAP_PROP_CONTRAST)))
print("CAP_PROP_SATURATION :'{}'".format(capture.get(cv2.CAP_PROP_SATURATION)))
print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
print("CAP_PROP_GAIN : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
print("CAP_PROP_EXPOSURE :'{}'".format(capture.get(cv2.CAP_PROP_EXPOSURE)))
print("CAP_PROP_CONVERT_RGB :'{}'".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))
print("CAP_PROP_RECTIFICATION :'{}'".format(capture.get(cv2.CAP_PROP_RECTIFICATION)))
print("CAP_PROP_ISO_SPEED :'{}'".format(capture.get(cv2.CAP_PROP_ISO_SPEED)))
print("CAP_PROP_BUFFERSIZE :'{}'".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)))

