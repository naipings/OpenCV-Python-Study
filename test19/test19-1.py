# 饱和运算

# 具体方法说明参见：https://blog.csdn.net/LOVEmy134611/article/details/120069198

import cv2
import numpy as np

x = np.uint8([250])
y = np.uint8([50])
# OpenCV中加法：250+50 = 300 => 255:
result_opencv = cv2.add(x, y)
print("cv2.add(x:'{}' , y:'{}') = '{}'".format(x, y, result_opencv))
# Numpy中加法：250+50 = 300 % 256 = 44:
result_numpy = x + y
print("x:'{}' + y:'{}' = '{}'".format(x, y, result_numpy))
