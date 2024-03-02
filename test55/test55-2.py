# 基于深度学习的人脸检测器
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 加载图片
image = cv2.imread('../opencvStudy/test55/imgs/test2.jpg')
# image = cv2.imread('../opencvStudy/test55/imgs/test1.jpg')
# image = cv2.imread('../opencvStudy/test55/imgs/test3.jpg')
# image = cv2.imread('../opencvStudy/test55/imgs/test4.jpg')
# image = cv2.imread('../opencvStudy/test55/imgs/test5.jpg')
# image = cv2.imread('../opencvStudy/test55/imgs/test6.jpg')
# image = cv2.imread('../opencvStudy/test55/imgs/test01.jpeg')
# image_2 = image.copy()
image_2 = cv2.imread('../opencvStudy/test55/imgs/test5.jpg')

# 我们使用预训练的深度学习人脸检测器模型执行人脸检测，使用i以下两种模型：
#   人脸检测器(FP16)：Caffe实现的浮点16版本，为了使用此检测器，需要下载模型文件及配置文件（见FP16文件夹）
#   人脸检测器(UINT8)：TensorFlow实现的8位量化版本，为了使用此检测器，需要下载模型文件及配置文件（见UINT8文件夹）

# 如何使用预训练的深度学习人脸检测器模型来检测人脸：
# 第一步，同样是加载预训练的模型：
# 加载预训练的模型， Caffe 实现的版本
net = cv2.dnn.readNetFromCaffe("../opencvStudy/test55/FP16/deploy.prototxt", "../opencvStudy/test55/FP16/res10_300x300_ssd_iter_140000_fp16.caffemodel")

# 为了获得最佳精度，必须分别对蓝色、绿色和红色通道执行（104, 177, 123）通道均值减法，并将图像调整为300*300的BGR图像，
# 在OpenCV中可以通过使用 cv2.dnn.blobFromImage() 函数进行此预处理：
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)

# 下一步是将blob设置为输入以获得结果，对整个网络执行前向计算以计算输出：
# 将 blob 设置为输入并获取检测结果
net.setInput(blob)
detections = net.forward()

# 最后一步是迭代检测并绘制结果，仅在相应置信度大于最小阈值时才将其可视化：
detected_faces = 0
w, h = image.shape[1], image.shape[0]
# 迭代所有检测结果
for i in range(0, detections.shape[2]):
    # 获取当前检测结果的置信度
    confidence = detections[0, 0, i, 2]
    # 如果置信大于最小置信度，则将其可视化
    if confidence > 0.7:
        detected_faces += 1
        # 获取当前检测结果的坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        # 绘制检测结果和置信度
        text = "{:.3f}%".format(confidence * 100)
        y = startY -10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


# 同理，对人脸检测器(UINT8)：
# 加载预训练的模型， Tensorflow 实现的版本：
net_2 = cv2.dnn.readNetFromTensorflow("../opencvStudy/test55/UINT8/opencv_face_detector_uint8.pb", "../opencvStudy/test55/UINT8/opencv_face_detector.pbtxt")
# 在OpenCV中可以通过使用 cv2.dnn.blobFromImage() 函数进行此预处理：
blob_2 = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)
# 将 blob 设置为输入并获取检测结果：
net_2.setInput(blob_2)
detections_2 = net_2.forward()
# 最后一步是迭代检测并绘制结果，仅在相应置信度大于最小阈值时才将其可视化：
detected_faces_2 = 0
w, h = image_2.shape[1], image_2.shape[0]
# 迭代所有检测结果
for i in range(0, detections_2.shape[2]):
    # 获取当前检测结果的置信度
    confidence = detections_2[0, 0, i, 2]
    # 如果置信大于最小置信度，则将其可视化
    if confidence > 0.7:
        detected_faces_2 += 1
        # 获取当前检测结果的坐标
        box = detections_2[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        # 绘制检测结果和置信度
        text = "{:.3f}%".format(confidence * 100)
        y = startY -10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image_2, (startX, startY), (endX, endY), (255, 0, 0), 3)
        cv2.putText(image_2, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face datection using OpenCV DNN face detector", fontsize=14, fontweight='bold')

show_img_with_matplotlib(image, "DNN face detector: " + str(detected_faces), 1)
show_img_with_matplotlib(image_2, "DNN face detector: " + str(detected_faces_2), 2)

plt.show()
