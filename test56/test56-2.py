# 基于CNN的人脸检测器
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import dlib

# 加载图像并转换为灰度图像
img = cv2.imread('../opencvStudy/test56/imgs/test17.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建CNN人脸检测器时，先将下载完成的预训练模型传递给此方法：
cnn_face_detector = dlib.cnn_face_detection_model_v1("../opencvStudy/test56/CNN-model/mmod_human_face_detector.dat")

# 之后，我们使用此检测器来检测人脸：
rects = cnn_face_detector(img, 0)

# show_detection() 用于显示检测结果：
def show_detection(image, faces):
    """使用矩形检测框显式标示每个检测到的人脸"""
    for face in faces:
        cv2.rectangle(image, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()), (255, 255, 0), 5)
    return image
# 绘制检测框
img_faces = show_detection(img.copy(), rects)

# 可视化
def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title, fontsize=8)
    plt.axis('off')
    plt.suptitle("Face datection using dlib CNN face detector", fontsize=14, fontweight='bold')

show_img_with_matplotlib(img_faces, "cnn_face_detector(img, 0): " + str(len(rects)), 1)
plt.show()
