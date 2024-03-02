# cv2.dnn.blobFromImage() 和 cv2.dnn.blobFromImages() 函数介绍
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# OpenCV 中深度学习为了执行前向计算，其输入应该是一个 blob，blob可以看作是经过预处理(包括缩放、裁剪、归一化、通道变换等)以馈送到网络的图像集合。
# 在OpenCV中，使用 cv2.dnn.blobFromImage() 构建 blob：
# 图像加载
image = cv2.imread("../opencvStudy/test68/imgs/test2.jpg")
# 利用 image 创建 4 维 blob
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)
# 代码意味着我们进行以下预处理：调整为300*300的BGR图像、分别对蓝色、绿色和红色通道执行(104.，117.，123.)均值减法。

# 接下来，我们可以将 blob 设置为输入并获得检测结果如下：
# 将 blob 设置为输入并获取检测结果
net = cv2.dnn.readNetFromCaffe("../opencvStudy/test68/FP16/deploy.prototxt", "../opencvStudy/test68/FP16/res10_300x300_ssd_iter_140000_fp16.caffemodel")
net.setInput(blob)
detections = net.forward()


# 接下来，首先介绍 cv2.dnn.blobFromImage() 和 cv2.dnn.blobFromImages() 函数。
retval = cv2.dnn.blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]])
# 此函数从 image 创建一个四维 blob，参数含义（见截屏）

# cv2.dnn.blobFromImages() 函数用法如下：
retval = cv2.dnn.blobFromImages(images[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]])
# 此函数可以从多个图像中创建一个四维 blob，通过这种方式，可以对整个网络执行一次前向计算获得多个图像的输出：
# 创建图像列表
images = []
for img in glob.glob('*.png'):
    images.append(cv2.imread(img))
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)
# 前向计算
net.setInput(blob_images)
detections = net.forward()


# cv2.dnn.blobFromImages() 和 cv2.dnn.blobFromImages() 的最后一个重要的参数是crop参数，它指示图像是否需要裁剪，
# 在 crop=true 的情况下，图像从中心进行裁剪。
# 为了更好地理解OpenCV在 cv2.dnn.blobFromImages() 和 cv2.dnn.blobFromImages() 函数中执行的裁剪，我们编写 get_cropped_img() 函数进行复刻：
def get_cropped_img(img):
    img_copy = img.copy()
    size = min(img_copy.shape[1], img_copy.shape[0])
    x1 = int(0.5 * (img_copy.shape[1] - size))
    y1 = int(0.5 * (img_copy.shape[0] - size))
    return img_copy[y1:(y1 + size), x1:(x1 + size)]
