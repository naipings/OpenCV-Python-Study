# 使用OpenCV检测面部特征点
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

# 加载图片
image = cv2.imread("../opencvStudy/test59/imgs/test1.jpg",0)
# image = cv2.imread("../opencvStudy/test59/imgs/test1.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
cas = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")
faces = cas.detectMultiScale(image , 1.5, 5)
# faces = cas.detectMultiScale(gray , 1.5, 5)
print("faces", faces)

# 创建特征点检测器并对其进行测试
print("testing LBF")

# 第一种面部特征点检测方法，第一行代码：
# 创建特征点检测器
facemark = cv2.face.createFacemarkLBF()

# 第一种面部特征点检测方法，第二行代码：
# 加载检测器模型
facemark.loadModel("../opencvStudy/test59/opencv_facemark/GSOC2017/data/lbfmodel.yaml")

# 第一种面部特征点检测方法，第三行代码：
# 检测面部特征点
ok, landmarks = facemark.fit(image , faces)
print ("landmarks LBF", ok, landmarks)


# 测试其他特征点检测算法
# print("testing AAM")
# facemark = cv2.face.createFacemarkAAM()
# facemark .loadModel("aam.xml")
# ok, landmarks = facemark.fit(image , faces)
# print ("landmarks AAM", ok, landmarks)


print("testing Kazemi")
facemark = cv2.face.createFacemarkKazemi()
facemark .loadModel("../opencvStudy/test59/opencv_facemark/face_landmark_model.dat")
ok, landmarks = facemark.fit(image , faces)
print ("landmarks Kazemi", ok, landmarks)
