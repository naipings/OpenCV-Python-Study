# 使用OpenCV进行人脸识别流程示例
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import glob

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

imgs = glob.glob('../opencvStudy/test64/train-imgs/*.jpg')
faces = []
rects = []

def detect_img(img, faces, rects):
    img = cv2.imread(img)
    (h, w) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    net = cv2.dnn.readNetFromCaffe("../opencvStudy/test64/FP16/deploy.prototxt", "../opencvStudy/test64/FP16/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104., 117., 123.], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        # 获取当前检测结果的置信度
        confidence = detections[0, 0, i, 2]
        # 如果置信大于最小置信度，则将其可视化
        if confidence > 0.7:
            # 获取当前检测结果的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            face_w = endX - startX
            face_h = endY - startY
            rects.append([startX, startY, face_w, face_h])
            faces.append(gray[startY:startY+face_h, startX:startX+face_w])
    return faces, rects
# 构造训练数据集
for img in imgs:
    faces, rects = detect_img(img, faces, rects)
# 根据实际情况构造标签数组
labels = [0] * len(faces)
# 训练识别器模型
face_recognizer.train(faces, np.array(labels))
# 加载测试图像并进行测试
img_test = cv2.imread('../opencvStudy/test64/imgs/test2.jpg')
face_test, rect_test = [], []
face_test, rect_test =  detect_img(img, face_test, rect_test)

label, confidence = face_recognizer.predict(face_test[0])
print(label, confidence)
