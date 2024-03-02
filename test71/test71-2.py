# 使用 YOLO V3 进行目标检测
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

# YOLO V3 使用一些技巧来改进训练和提高性能，包括多尺度预测和更好的主干分类器等。

# 加载类别名
class_names = open('coco.names').read().strip().split('\n')
# 加载网络及参数
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# 加载测试图像
image = cv2.imread('test_img.jpg')
(H, W) = image.shape[:2]
# 获取网络输出
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# 预处理
blob = cv2.dnn.blobFromImage(
    image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
print(blob.shape)
# 前向计算
net.setInput(blob)
layerOutputs = net.forward(layer_names)
t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))
# 构建结果数组
boxes = []
confidences = []
class_ids = []
# 循环输出结果
for output in layerOutputs:
    # 循环检测结果
    for detection in output:
        # 获取类别 id 和置信度
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # 过滤低置信度目标
        if confidence > 0.25:
            # 使用原始图像的尺寸缩放边界框坐标
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # 计算边界框左上角坐标
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # 将结果添加到结果数组中
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
# 应用非极大值抑制
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
# 绘制结果
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = "{}: {:.4f}".format(class_names[class_ids[i]], confidences[i])
        labelSize, baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        y = max(y, labelSize[1])
        cv2.rectangle(
            image, (x, y - labelSize[1]), (x + labelSize[0], y + 0), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
