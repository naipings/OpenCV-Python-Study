# 使用 MobileNet-SSD 进行目标检测
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# MobileNets 是用于移动视觉应用的高效卷积神经网络，
# MobileNet-SSD 在 COCO 数据集上进行了训练，达到了 77.27%mAP，可用于检测到20种对象类别：
# Person(人): Person(人)
# Animal(动物): Bird(鸟), cat(猫), cow(牛), dog(狗), horse(马), sheep(羊)
# Vehicle(交通工具): Aeroplane(飞机), bicycle(自行车), boat(船), bus(公共汽车), car(小轿车), motorbike(摩托车), train(火车)
# Indoor(室内): Bottle(水瓶), chair(椅子), dining table(餐桌), potted plant(盆栽), sofa(沙发), TV/monitor(电视/显示器)

# 补充：COCO：coco是一个缩写，它的完整形式是“ConvolutionalNeuralNetworksforVisualRecognition”，中文翻译为“卷积神经网络视觉识别”，简称CNN。
# 它是一种深度学习技术，用于图像识别和分类，广泛应用于计算机视觉领域的各种任务，如图像分类、目标检测、语义分割、图像生成等。

# 通过使用 MobileNet-SSD 和 Caffe 预训练模型，使用 OpenCV DNN 模块执行对象检测：
# 加载模型及参数
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
# 图片读取
image = cv2.imread('test_img.jpg')
# 定义类别名
class_names = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
               11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
# 预处理
blob = cv2.dnn.blobFromImage(
    image, 0.007843, (300, 300), (127.5, 127.5, 127.5))
print(blob.shape)
# 前向计算
net.setInput(blob)
detections = net.forward()

t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))
# 输入图像尺寸
dim = 300

# 处理检测结果
for i in range(detections.shape[2]):
    # 获得预测的置信度
    confidence = detections[0, 0, i, 2]

    # 去除置信度较低的预测
    if confidence > 0.2:
        # 获取类别标签
        class_id = int(detections[0, 0, i, 1])
        # 获取检测到目标对象框的坐标
        xLeftBottom = int(detections[0, 0, i, 3] * dim)
        yLeftBottom = int(detections[0, 0, i, 4] * dim)
        xRightTop = int(detections[0, 0, i, 5] * dim)
        yRightTop = int(detections[0, 0, i, 6] * dim)
        # 缩放比例系数
        heightFactor = image.shape[0] / dim
        widthFactor = image.shape[1] / dim
        # 根据缩放比例系数计算检测结果最终坐标
        xLeftBottom = int(widthFactor * xLeftBottom)
        yLeftBottom = int(heightFactor * yLeftBottom)
        xRightTop = int(widthFactor * xRightTop)
        yRightTop = int(heightFactor * yRightTop)
        # 绘制矩形框
        cv2.rectangle(image, (xLeftBottom, yLeftBottom),
                      (xRightTop, yRightTop), (0, 255, 0), 2)
        # 绘制置信度和类别
        if class_id in class_names:
            label = class_names[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                          (xLeftBottom + labelSize[0], yLeftBottom + 0), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
